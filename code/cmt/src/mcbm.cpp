#include "mcbm.h"
#include "utils.h"
#include "lbfgs.h"

#include "Eigen/Core"
using Eigen::Matrix;
using Eigen::Map;

#include <vector>
using std::make_pair;

#include <cmath>
using std::min;

#include <iostream>
using std::cout;
using std::endl;

#include <iomanip>
using std::setw;
using std::setprecision;

struct InstanceLBFGS {
	const MCBM* mcbm;
	const MCBM::Parameters* params;
	const MatrixXd* input;
	const MatrixXd* output;
};

typedef Map<Matrix<lbfgsfloatval_t, Dynamic, Dynamic> > MatrixLBFGS;
typedef Map<Matrix<lbfgsfloatval_t, Dynamic, 1> > VectorLBFGS;

static int callbackLBFGS(
	void *instance,
	const lbfgsfloatval_t *x,
	const lbfgsfloatval_t *g,
	const lbfgsfloatval_t fx,
	const lbfgsfloatval_t xnorm,
	const lbfgsfloatval_t gnorm,
	const lbfgsfloatval_t, int,
	int iteration,
	int)
{
	// unpack user data
	const InstanceLBFGS& inst = *static_cast<InstanceLBFGS*>(instance);
	const MCBM& mcbm = *inst.mcbm;
	const MCBM::Parameters& params = *inst.params;

	if(params.verbosity > 0)
		cout << setw(6) << iteration << setw(10) << setprecision(5) << fx << endl;

	if(params.callback && iteration % params.cbIter == 0) {
		// TODO: fix this nasty hack
		const_cast<MCBM&>(mcbm).setParameters(x, params);

		if(!(*params.callback)(iteration, mcbm))
			return 1;
	}

	return 0;
}



static lbfgsfloatval_t evaluateLBFGS(
	void* instance,
	const lbfgsfloatval_t* x,
	lbfgsfloatval_t* g,
	int, double)
{
	// unpack user data
	const InstanceLBFGS& inst = *static_cast<InstanceLBFGS*>(instance);
	const MCBM& mcbm = *inst.mcbm;
	const MCBM::Parameters& params = *inst.params;
	const MatrixXd& input = *inst.input;
	const MatrixXd& output = *inst.output;

	return mcbm.computeGradient(input, output, x, g, params);
}



MCBM::Parameters::Parameters() : 
	ConditionalDistribution::Parameters::Parameters()
{
	trainPriors = true;
	trainWeights = true;
	trainFeatures = true;
	trainPredictors = true;
	trainInputBias = true;
	trainOutputBias = true;
	regularizeFeatures = 0.;
	regularizePredictors = 0.;
}



MCBM::Parameters::Parameters(const Parameters& params) :
	ConditionalDistribution::Parameters::Parameters(params),
	trainPriors(params.trainPriors),
	trainWeights(params.trainWeights),
	trainFeatures(params.trainFeatures),
	trainPredictors(params.trainPredictors),
	trainInputBias(params.trainInputBias),
	trainOutputBias(params.trainOutputBias),
	regularizeFeatures(params.regularizeFeatures),
	regularizePredictors(params.regularizePredictors)
{
	if(params.callback)
		callback = params.callback->copy();
}



MCBM::Parameters::~Parameters() {
}



MCBM::Parameters& MCBM::Parameters::operator=(const Parameters& params) {
	ConditionalDistribution::Parameters::operator=(params);

	trainPriors = params.trainPriors;
	trainWeights = params.trainWeights;
	trainFeatures = params.trainFeatures;
	trainPredictors = params.trainPredictors;
	trainInputBias = params.trainInputBias;
	trainOutputBias = params.trainOutputBias;
	regularizeFeatures = params.regularizeFeatures;
	regularizePredictors = params.regularizePredictors;

	return *this;
}



MCBM::MCBM(int dimIn, int numComponents, int numFeatures) : 
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



MCBM::MCBM(int dimIn, const MCBM& mcbm) : 
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



MCBM::~MCBM() {
}



MatrixXd MCBM::sample(const MatrixXd& input) const {
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

	return (Array<double, 1, Dynamic>::Random(input.cols()).abs() < logProb1.exp()).cast<double>();
}



Array<double, 1, Dynamic> MCBM::logLikelihood(const MatrixXd& input, const MatrixXd& output) const {
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
}



bool MCBM::train(const MatrixXd& input, const MatrixXd& output, const Parameters& params) {
	if(input.rows() != mDimIn || output.rows() != 1)
		throw Exception("Data has wrong dimensionality.");
	if(input.cols() != output.cols())
		throw Exception("The number of inputs and outputs should be the same.");

	// copy parameters for L-BFGS
	lbfgsfloatval_t* x = parameters(params);

	// optimization hyperparameters
	lbfgs_parameter_t hyperparams;
	lbfgs_parameter_init(&hyperparams);
	hyperparams.max_iterations = params.maxIter;
	hyperparams.m = params.numGrad;
	hyperparams.epsilon = params.threshold;
	hyperparams.linesearch = LBFGS_LINESEARCH_MORETHUENTE;
 	hyperparams.max_linesearch = 100;
 	hyperparams.ftol = 1e-4;
 	hyperparams.xtol = 1e-32;

	// wrap additional arguments
	InstanceLBFGS instance = { this, &params, &input, &output };

	// start LBFGS optimization
	int status = LBFGSERR_MAXIMUMITERATION;
	if(params.maxIter > 0)
		status = lbfgs(numParameters(params), x, 0, &evaluateLBFGS, &callbackLBFGS, &instance, &hyperparams);

	// copy parameters back
	setParameters(x, params);

	// free memory used by LBFGS
	lbfgs_free(x);

	if(status >= 0) {
		return true;
	} else {
		if(status != LBFGSERR_MAXIMUMITERATION)
			cout << "There seems to be something not quite right with the optimization (" << status << ")." << endl;
		return false;
	}
}



lbfgsfloatval_t* MCBM::parameters(const Parameters& params) const {
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



void MCBM::setParameters(const lbfgsfloatval_t* x, const Parameters& params) {
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



double MCBM::computeGradient(
	const MatrixXd& inputCompl,
	const MatrixXd& outputCompl,
	const lbfgsfloatval_t* x,
	lbfgsfloatval_t* g,
	const Parameters& params) const
{
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
	int batchSize = min(params.batchSize, numData);

	for(int b = 0; b < inputCompl.cols(); b += batchSize) {
		// TODO: copying memory necessary?
		const MatrixXd input = inputCompl.middleCols(b, min(batchSize, numData - b));
		const MatrixXd output = outputCompl.middleCols(b, min(batchSize, numData - b));

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

		logLik += (output.array() * logProb1 + (1. - output.array()) * logProb0).sum();

		if(!g)
			// don't compute gradients
			continue;

		Array<double, 1, Dynamic> tmp = output.array() * logProb0.exp() - (1. - output.array()) * logProb1.exp();

		ArrayXXd post0Tmp = logPost0.exp().rowwise() * tmp;
		ArrayXXd post1Tmp = logPost1.exp().rowwise() * tmp;
		ArrayXXd postDiffTmp = post1Tmp - post0Tmp;

		if(params.trainPriors)
			priorsGrad -= postDiffTmp.rowwise().sum().matrix();

		if(params.trainWeights)
			weightsGrad -= postDiffTmp.matrix() * featureOutputSq.transpose();

		if(params.trainFeatures) {
			ArrayXXd tmp2 = weights.transpose() * postDiffTmp.matrix() * 2.;
			MatrixXd tmp3 = featureOutput * tmp2;
			featuresGrad -= input * tmp3.transpose();
		}

		if(params.trainPredictors)
			predictorsGrad -= post1Tmp.matrix() * input.transpose();

		if(params.trainInputBias)
			inputBiasGrad -= input * postDiffTmp.matrix().transpose();

		if(params.trainOutputBias)
			outputBiasGrad -= post1Tmp.rowwise().sum().matrix();
	}

	double normConst = inputCompl.cols() / log(2.);

	if(g)
		for(int i = 0; i < offset; ++i)
			g[i] /= normConst;

	return -logLik / normConst;
}



double MCBM::checkGradient(
	const MatrixXd& input,
	const MatrixXd& output,
	double epsilon,
	const Parameters& params) const
{
	if(input.rows() != mDimIn || output.rows() != 1)
		throw Exception("Data has wrong dimensionality.");
	if(input.cols() != output.cols())
		throw Exception("The number of inputs and outputs should be the same.");

	// request memory for LBFGS and copy parameters
	lbfgsfloatval_t* x = parameters(params);

	int numParams = numParameters(params);

	lbfgsfloatval_t y[numParams];
	lbfgsfloatval_t g[numParams];
	lbfgsfloatval_t n[numParams];
	lbfgsfloatval_t val1;
	lbfgsfloatval_t val2;

	// make another copy
	for(int i = 0; i < numParams; ++i)
		y[i] = x[i];

	// arguments to LBFGS function
	InstanceLBFGS instance = { this, &params, &input, &output };

	// compute numerical gradient using central differences
	for(int i = 0; i < numParams; ++i) {
		y[i] = x[i] + epsilon;
		val1 = evaluateLBFGS(&instance, y, 0, 0, 0.);
		y[i] = x[i] - epsilon;
		val2 = evaluateLBFGS(&instance, y, 0, 0, 0.);
		y[i] = x[i];
		n[i] = (val1 - val2) / (2. * epsilon);
	}

	// compute analytical gradient
	evaluateLBFGS(&instance, x, g, 0, 0.);

	// squared error
	double err = 0.;
	for(int i = 0; i < numParams; ++i)
		err += (g[i] - n[i]) * (g[i] - n[i]);

	return sqrt(err);
}



pair<pair<ArrayXXd, ArrayXXd>, Array<double, 1, Dynamic> > MCBM::computeDataGradient(
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
