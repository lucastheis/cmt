#include "mcbm.h"
#include "utils.h"

#include <utility>
using std::pair;
using std::make_pair;

#include <cmath>
using std::min;
using std::max;
using std::exp;
using std::log;

#include "Eigen/Core"
using Eigen::Dynamic;
using Eigen::Array;
using Eigen::ArrayXXd;
using Eigen::MatrixXd;
using Eigen::VectorXd;

CMT::MCBM::Parameters::Parameters() :
	Trainable::Parameters::Parameters(),
	trainPriors(true),
	trainWeights(true),
	trainFeatures(true),
	trainPredictors(true),
	trainInputBias(true),
	trainOutputBias(true),
	regularizeFeatures(0.),
	regularizePredictors(0.),
	regularizeWeights(0.)
{
}



CMT::MCBM::Parameters::Parameters(const Parameters& params) :
	Trainable::Parameters::Parameters(params),
	trainPriors(params.trainPriors),
	trainWeights(params.trainWeights),
	trainFeatures(params.trainFeatures),
	trainPredictors(params.trainPredictors),
	trainInputBias(params.trainInputBias),
	trainOutputBias(params.trainOutputBias),
	regularizeFeatures(params.regularizeFeatures),
	regularizePredictors(params.regularizePredictors),
	regularizeWeights(params.regularizeWeights)
{
}



CMT::MCBM::Parameters& CMT::MCBM::Parameters::operator=(const Parameters& params) {
	Trainable::Parameters::operator=(params);

	trainPriors = params.trainPriors;
	trainWeights = params.trainWeights;
	trainFeatures = params.trainFeatures;
	trainPredictors = params.trainPredictors;
	trainInputBias = params.trainInputBias;
	trainOutputBias = params.trainOutputBias;
	regularizeFeatures = params.regularizeFeatures;
	regularizePredictors = params.regularizePredictors;
	regularizeWeights = params.regularizeWeights;

	return *this;
}



CMT::MCBM::MCBM(int dimIn, int numComponents, int numFeatures) :
	mDimIn(dimIn),
	mNumComponents(numComponents),
	mNumFeatures(numFeatures < 0 ? dimIn : numFeatures)
{
	// check hyperparameters
	if(mNumComponents < 1)
		throw Exception("The number of components has to be positive.");

	// initialize parameters
	mPriors = VectorXd::Zero(mNumComponents);
	mWeights = ArrayXXd::Random(mNumComponents, mNumFeatures).abs() / 100. + 0.01;
	mFeatures = sampleNormal(mDimIn, mNumFeatures) / 100.;
	mPredictors = sampleNormal(mNumComponents, mDimIn) / 100.;
	mInputBias = MatrixXd::Zero(mDimIn, mNumComponents);
	mOutputBias = VectorXd::Zero(mNumComponents);
}



CMT::MCBM::MCBM(int dimIn, const MCBM& mcbm) : 
	mDimIn(dimIn),
	mNumComponents(mcbm.numComponents()),
	mNumFeatures(mcbm.numFeatures())
{
	// initialize parameters
	mPriors = VectorXd::Zero(mNumComponents);
	mWeights = ArrayXXd::Random(mNumComponents, mNumFeatures).abs() / 100. + 0.01;
	mFeatures = sampleNormal(mDimIn, mNumFeatures) / 100.;
	mPredictors = sampleNormal(mNumComponents, mDimIn) / 100.;
	mInputBias = MatrixXd::Zero(mDimIn, mNumComponents);
	mOutputBias = VectorXd::Zero(mNumComponents);
}



CMT::MCBM::~MCBM() {
}



MatrixXd CMT::MCBM::sample(const MatrixXd& input) const {
	if(mDimIn) {
		// some intermediate computations
		ArrayXXd featureEnergy = mWeights * (mFeatures.transpose() * input).array().square().matrix();
		ArrayXXd biasEnergy = mInputBias.transpose() * input;
		ArrayXXd predictorEnergy = mPredictors * input;

		// unnormalized probabilities of generating a 0 or 1 for each component
		ArrayXXd logProb0 = (featureEnergy + biasEnergy).colwise() + mPriors.array();
		ArrayXXd logProb1 = (logProb0 + predictorEnergy).colwise() + mOutputBias.array();

		// sum over components
		logProb0 = logSumExp(logProb0);
		logProb1 = logSumExp(logProb1);

		// stack row vectors
		ArrayXXd logProb01(2, input.cols());
		logProb01 << logProb0, logProb1; 

		// normalize log-probability
		logProb1 -= logSumExp(logProb01);

		ArrayXXd uniRand = Array<double, 1, Dynamic>::Random(input.cols()).abs();
		return (uniRand < logProb1.exp()).cast<double>();
	} else {
		// input is zero-dimensional
		double logProb0 = logSumExp(mPriors)[0];
		double logProb1 = logSumExp(mPriors + mOutputBias)[0];

		ArrayXXd logProb01(2, 1);
		logProb01 << logProb0, logProb1;

		logProb1 -= logSumExp(logProb01)[0];

		return (
			Array<double, 1, Dynamic>::Random(input.cols()).abs() <
			Array<double, 1, Dynamic>::Zero(input.cols()) + exp(logProb1)).cast<double>();
	}
}



Array<int, 1, Dynamic> CMT::MCBM::samplePrior(const MatrixXd& input) const {
	if(input.rows() != dimIn())
		throw Exception("Inputs have wrong dimensionality.");

	ArrayXXd featureEnergy = mWeights * (mFeatures.transpose() * input).array().square().matrix();
	ArrayXXd biasEnergy = mInputBias.transpose() * input;

	ArrayXXd predictorEnergy = mPredictors * input;

	ArrayXXd tmp0 = (featureEnergy + biasEnergy).colwise() + mPriors.array();
	ArrayXXd tmp1 = (tmp0 + predictorEnergy).colwise() + mOutputBias.array();

	ArrayXXd logPrior = tmp0 + tmp1;
	logPrior.rowwise() -= logSumExp(logPrior);

	ArrayXXd prior = logPrior.exp();

	Array<int, 1, Dynamic> labels(input.cols());

	#pragma omp parallel for
	for(int j = 0; j < input.cols(); ++j) {
		int i = 0;
		double urand = static_cast<double>(rand()) / (static_cast<long>(RAND_MAX) + 1l);
		double cdf;

		// compute index
		for(cdf = prior(0, j); cdf < urand; cdf += prior(i, j))
			++i;

		labels[j] = i;
	}

	return labels;
}



Array<int, 1, Dynamic> CMT::MCBM::samplePosterior(
	const MatrixXd& input,
	const MatrixXd& output) const
{
	if(output.cols() != input.cols())
		throw Exception("Number of inputs and outputs must be the same.");
	if(input.rows() != dimIn())
		throw Exception("Inputs have wrong dimensionality.");
	if(output.rows() != dimOut())
		throw Exception("Outputs have wrong dimensionality.");

	ArrayXXd featureEnergy = mWeights * (mFeatures.transpose() * input).array().square().matrix();
	ArrayXXd biasEnergy = mInputBias.transpose() * input;

	ArrayXXd predictorEnergy = mPredictors * input;

	ArrayXXd tmp0 = (featureEnergy + biasEnergy).colwise() + mPriors.array();
	ArrayXXd tmp1 = (tmp0 + predictorEnergy).colwise() + mOutputBias.array();

	ArrayXXd logPosterior = 
		tmp0.rowwise() * (1. - output.row(0).array()) +
		tmp1.rowwise() * output.row(0).array();
	logPosterior.rowwise() -= logSumExp(logPosterior);

	ArrayXXd post = logPosterior.exp();

	Array<int, 1, Dynamic> labels(input.cols());

	#pragma omp parallel for
	for(int j = 0; j < input.cols(); ++j) {
		int i = 0;
		double urand = static_cast<double>(rand()) / (static_cast<long>(RAND_MAX) + 1l);
		double cdf;

		// compute index
		for(cdf = post(0, j); cdf < urand; cdf += post(i, j))
			++i;

		labels[j] = i;
	}

	return labels;
}



Array<double, 1, Dynamic> CMT::MCBM::logLikelihood(
	const MatrixXd& input,
	const MatrixXd& output) const 
{
	if(input.rows() != dimIn())
		throw Exception("Input has wrong dimensionality.");
	if(output.rows() != dimOut())
		throw Exception("Input has wrong dimensionality.");
	if(input.cols() != output.cols())
		throw Exception("The number of inputs and outputs must be the same.");

	if(mDimIn) {
		// some intermediate computations
		ArrayXXd featureEnergy = mWeights * (mFeatures.transpose() * input).array().square().matrix();
		ArrayXXd biasEnergy = mInputBias.transpose() * input;
		ArrayXXd predictorEnergy = mPredictors * input;

		// unnormalized probabilities of generating a 0 or 1 for each component
		ArrayXXd logProb0 = (featureEnergy + biasEnergy).colwise() + mPriors.array();
		ArrayXXd logProb1 = (logProb0 + predictorEnergy).colwise() + mOutputBias.array();

		// sum over components
		logProb0 = logSumExp(logProb0);
		logProb1 = logSumExp(logProb1);

		// stack row vectors
		ArrayXXd logProb01(2, input.cols());
		logProb01 << logProb0, logProb1; 

		// normalized log-probabilities
		Array<double, 1, Dynamic> logNorm = logSumExp(logProb01);
		logProb1 -= logNorm;
		logProb0 -= logNorm;

		return output.array() * logProb1 + (1. - output.array()) * logProb0;
	} else {
		// input is zero-dimensional
		double logProb0 = logSumExp(mPriors)[0];
		double logProb1 = logSumExp(mPriors + mOutputBias)[0];

		ArrayXXd logProb01(2, 1);
		logProb01 << logProb0, logProb1;

		double logNorm = logSumExp(logProb01)[0];
		logProb0 -= logNorm;
		logProb1 -= logNorm;

		return output.array() * logProb1 + (1. - output.array()) * logProb0;
	}
}



int CMT::MCBM::numParameters(const Trainable::Parameters& params_) const {
	const Parameters& params = dynamic_cast<const Parameters&>(params_);

	int numParams = 0;
	if(params.trainPriors)
		numParams += mPriors.size();
	if(params.trainWeights)
		numParams += mWeights.size();
	if(params.trainFeatures)
		numParams += mFeatures.size();
	if(params.trainPredictors)
		numParams += mPredictors.size();
	if(params.trainInputBias)
		numParams += mInputBias.size();
	if(params.trainOutputBias)
		numParams += mOutputBias.size();
	return numParams;
}



lbfgsfloatval_t* CMT::MCBM::parameters(const Trainable::Parameters& params_) const {
	const Parameters& params = dynamic_cast<const Parameters&>(params_);

	lbfgsfloatval_t* x = lbfgs_malloc(numParameters(params));

	int k = 0;
	if(params.trainPriors)
		for(int i = 0; i < mPriors.size(); ++i, ++k)
			x[k] = mPriors.data()[i];
	if(params.trainWeights)
		for(int i = 0; i < mWeights.size(); ++i, ++k)
			x[k] = mWeights.data()[i];
	if(params.trainFeatures)
		for(int i = 0; i < mFeatures.size(); ++i, ++k)
			x[k] = mFeatures.data()[i];
	if(params.trainPredictors)
		for(int i = 0; i < mPredictors.size(); ++i, ++k)
			x[k] = mPredictors.data()[i];
	if(params.trainInputBias)
		for(int i = 0; i < mInputBias.size(); ++i, ++k)
			x[k] = mInputBias.data()[i];
	if(params.trainOutputBias)
		for(int i = 0; i < mOutputBias.size(); ++i, ++k)
			x[k] = mOutputBias.data()[i];

	return x;
}



void CMT::MCBM::setParameters(const lbfgsfloatval_t* x, const Trainable::Parameters& params_) {
	const Parameters& params = dynamic_cast<const Parameters&>(params_);

	int offset = 0;

	if(params.trainPriors) {
		mPriors = VectorLBFGS(const_cast<double*>(x), mNumComponents);
		offset += mPriors.size();
	}

	if(params.trainWeights) {
		mWeights = MatrixLBFGS(const_cast<double*>(x) + offset, mNumComponents, mNumFeatures);
		offset += mWeights.size();
	}

	if(params.trainFeatures) {
		mFeatures = MatrixLBFGS(const_cast<double*>(x) + offset, mDimIn, mNumFeatures);
		offset += mFeatures.size();
	}

	if(params.trainPredictors) {
		mPredictors = MatrixLBFGS(const_cast<double*>(x) + offset, mNumComponents, mDimIn);
		offset += mPredictors.size();
	}

	if(params.trainInputBias) {
		mInputBias = MatrixLBFGS(const_cast<double*>(x) + offset, mDimIn, mNumComponents);
		offset += mInputBias.size();
	}

	if(params.trainOutputBias) {
		mOutputBias = VectorLBFGS(const_cast<double*>(x) + offset, mNumComponents);
		offset += mOutputBias.size();
	}
}



double CMT::MCBM::parameterGradient(
	const MatrixXd& inputCompl,
	const MatrixXd& outputCompl,
	const lbfgsfloatval_t* x,
	lbfgsfloatval_t* g,
	const Trainable::Parameters& params_) const
{
	const Parameters& params = dynamic_cast<const Parameters&>(params_);

	// average log-likelihood
	double logLik = 0.;

	// interpret memory for parameters and gradients
	lbfgsfloatval_t* y = const_cast<lbfgsfloatval_t*>(x);

	int offset = 0;

	VectorLBFGS priors(params.trainPriors ? y : const_cast<double*>(mPriors.data()), mNumComponents);
	VectorLBFGS priorsGrad(g, mNumComponents);
	if(params.trainPriors)
		offset += priors.size();

	MatrixLBFGS weights(params.trainWeights ? y + offset : const_cast<double*>(mWeights.data()), mNumComponents, mNumFeatures);
	MatrixLBFGS weightsGrad(g + offset, mNumComponents, mNumFeatures);
	if(params.trainWeights)
		offset += weights.size();

	MatrixLBFGS features(params.trainFeatures ? y + offset : const_cast<double*>(mFeatures.data()), mDimIn, mNumFeatures);
	MatrixLBFGS featuresGrad(g + offset, mDimIn, mNumFeatures);
	if(params.trainFeatures)
		offset += features.size();

	MatrixLBFGS predictors(params.trainPredictors ? y + offset : const_cast<double*>(mPredictors.data()), mNumComponents, mDimIn);
	MatrixLBFGS predictorsGrad(g + offset, mNumComponents, mDimIn);
	if(params.trainPredictors)
		offset += predictors.size();

	MatrixLBFGS inputBias(params.trainInputBias ? y + offset : const_cast<double*>(mInputBias.data()), mDimIn, mNumComponents);
	MatrixLBFGS inputBiasGrad(g + offset, mDimIn, mNumComponents);
	if(params.trainInputBias)
		offset += inputBias.size();

	VectorLBFGS outputBias(params.trainOutputBias ? y + offset : const_cast<double*>(mOutputBias.data()), mNumComponents);
	VectorLBFGS outputBiasGrad(g + offset, mNumComponents);
	if(params.trainOutputBias)
		offset += outputBias.size();

	if(g) {
		// initialize gradients
		if(params.trainPriors)
			priorsGrad.setZero();
		if(params.trainWeights)
			weightsGrad.setZero();
		if(params.trainFeatures)
			featuresGrad.setZero();
		if(params.trainPredictors)
			predictorsGrad.setZero();
		if(params.trainInputBias)
			inputBiasGrad.setZero();
		if(params.trainOutputBias)
			outputBiasGrad.setZero();
	}

	// split data into batches for better performance
	int numData = static_cast<int>(inputCompl.cols());
	int batchSize = min(max(params.batchSize, 10), numData);

	#pragma omp parallel for
	for(int b = 0; b < inputCompl.cols(); b += batchSize) {
		const MatrixXd& input = inputCompl.middleCols(b, min(batchSize, numData - b));
		const MatrixXd& output = outputCompl.middleCols(b, min(batchSize, numData - b));

		ArrayXXd featureOutput = features.transpose() * input;
		MatrixXd featureOutputSq = featureOutput.square();
		MatrixXd weightsOutput = weights * featureOutputSq;
		ArrayXXd predictorOutput = predictors * input;

		// unnormalized posteriors over components for both possible outputs
		ArrayXXd logPost0 = (weightsOutput + inputBias.transpose() * input).colwise() + priors;
		ArrayXXd logPost1 = (logPost0 + predictorOutput).colwise() + outputBias.array();

		// sum over components to get unnormalized probabilities of outputs
		Array<double, 1, Dynamic> logProb0 = logSumExp(logPost0);
		Array<double, 1, Dynamic> logProb1 = logSumExp(logPost1);
	
		// normalize posteriors over components
		logPost0.rowwise() -= logProb0;
		logPost1.rowwise() -= logProb1;

		// stack row vectors
		ArrayXXd logProb01(2, input.cols());
		logProb01 << logProb0, logProb1; 

		// normalize log-probabilities
		Array<double, 1, Dynamic> logNorm = logSumExp(logProb01);
		logProb1 -= logNorm;
		logProb0 -= logNorm;

		double logLikBatch = (output.array() * logProb1 + (1. - output.array()) * logProb0).sum();

		#pragma omp critical
		logLik += logLikBatch;

		if(!g)
			// don't compute gradients
			continue;

		Array<double, 1, Dynamic> tmp = output.array() * logProb0.exp() - (1. - output.array()) * logProb1.exp();

		ArrayXXd post0Tmp = logPost0.exp().rowwise() * tmp;
		ArrayXXd post1Tmp = logPost1.exp().rowwise() * tmp;
		ArrayXXd postDiffTmp = post1Tmp - post0Tmp;

		// update gradients
		if(params.trainPriors)
			#pragma omp critical
			priorsGrad -= postDiffTmp.rowwise().sum().matrix();

		if(params.trainWeights)
			#pragma omp critical
			weightsGrad -= postDiffTmp.matrix() * featureOutputSq.transpose();

		if(params.trainFeatures) {
			ArrayXXd tmp2 = weights.transpose() * postDiffTmp.matrix() * 2.;
			MatrixXd tmp3 = featureOutput * tmp2;
			#pragma omp critical
			featuresGrad -= input * tmp3.transpose();
		}

		if(params.trainPredictors)
			#pragma omp critical
			predictorsGrad -= post1Tmp.matrix() * input.transpose();

		if(params.trainInputBias)
			#pragma omp critical
			inputBiasGrad -= input * postDiffTmp.matrix().transpose();

		if(params.trainOutputBias)
			#pragma omp critical
			outputBiasGrad -= post1Tmp.rowwise().sum().matrix();
	}

	double normConst = inputCompl.cols() * log(2.) * dimOut();

	if(g) {
		for(int i = 0; i < offset; ++i)
			g[i] /= normConst;

		if(params.trainFeatures)
			featuresGrad += params.regularizeFeatures.gradient(features);

		if(params.trainPredictors)
			predictorsGrad += params.regularizePredictors.gradient(predictors.transpose()).transpose();

		if(params.trainWeights)
			weightsGrad += params.regularizeWeights.gradient(weights);
	}

	double value = -logLik / normConst;

	if(params.trainFeatures)
		value += params.regularizeFeatures.evaluate(features);

	if(params.trainPredictors)
		value += params.regularizePredictors.evaluate(predictors.transpose());

	if(params.trainWeights)
		value += params.regularizeWeights.evaluate(weights);

	return value;
}



pair<pair<ArrayXXd, ArrayXXd>, Array<double, 1, Dynamic> > CMT::MCBM::computeDataGradient(
	const MatrixXd& input,
	const MatrixXd& output) const
{
	throw Exception("Not implemented.");

	return make_pair(
		make_pair(
			ArrayXXd::Zero(input.rows(), input.cols()),
			ArrayXXd::Zero(output.rows(), output.cols())), 
		logLikelihood(input, output));
}



bool CMT::MCBM::train(
	const MatrixXd& input,
	const MatrixXd& output,
	const MatrixXd* inputVal,
	const MatrixXd* outputVal,
	const Trainable::Parameters& params)
{
	if(!mDimIn) {
		// zero-dimensional inputs; MCBM reduces to Bernoulli
		double prob = output.array().mean();
		mPriors.setZero();
		mOutputBias.setConstant(prob > 0. ? log(prob) : -50.);
		return true;
	} else {
		return Trainable::train(input, output, inputVal, outputVal, params);
	}
}
