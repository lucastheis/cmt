#include "stm.h"
#include "glm.h"
#include "utils.h"
#include "Eigen/Cholesky"
#include "Eigen/LU"

#include "Eigen/Eigenvalues"
using Eigen::SelfAdjointEigenSolver;

#include "Eigen/Core"
using Eigen::MatrixXd;
using Eigen::ArrayXXd;
using Eigen::ArrayXd;
using Eigen::Array;
using Eigen::Dynamic;
using Eigen::VectorXi;
using Eigen::VectorXd;

#include <utility>
using std::pair;
using std::make_pair;

#include <cmath>
using std::max;
using std::min;

CMT::STM::Parameters::Parameters() :
	Trainable::Parameters::Parameters(),
	trainBiases(true),
	trainWeights(true),
	trainFeatures(true),
	trainPredictors(true),
	trainLinearPredictor(true),
	regularizeWeights(0.),
	regularizeFeatures(0.),
	regularizePredictors(0.),
	regularizeLinearPredictor(0.),
	regularizer(L1)
{
}



CMT::STM::Parameters::Parameters(const Parameters& params) :
	Trainable::Parameters::Parameters(params),
	trainBiases(params.trainBiases),
	trainWeights(params.trainWeights),
	trainFeatures(params.trainFeatures),
	trainPredictors(params.trainPredictors),
	trainLinearPredictor(params.trainLinearPredictor),
	regularizeWeights(params.regularizeWeights),
	regularizeFeatures(params.regularizeFeatures),
	regularizePredictors(params.regularizePredictors),
	regularizeLinearPredictor(params.regularizeLinearPredictor),
	regularizer(params.regularizer)
{
}



CMT::STM::Parameters& CMT::STM::Parameters::operator=(const Parameters& params) {
	Trainable::Parameters::operator=(params);

	trainBiases = params.trainBiases;
	trainWeights = params.trainWeights;
	trainFeatures = params.trainFeatures;
	trainPredictors = params.trainPredictors;
	trainLinearPredictor = params.trainLinearPredictor;
	regularizeWeights = params.regularizeWeights;
	regularizeFeatures = params.regularizeFeatures;
	regularizePredictors = params.regularizePredictors;
	regularizeLinearPredictor = params.regularizeLinearPredictor;
	regularizer = params.regularizer;

	return *this;
}



CMT::STM::STM(int dimIn, int numComponents, int numFeatures) :
	mDimInNonlinear(dimIn),
	mDimInLinear(0),
	mNumComponents(numComponents),
	mNumFeatures(numFeatures < 0 ? dimIn : numFeatures)
{
	// check hyperparameters
	if(mDimInNonlinear < 0)
		throw Exception("The input dimensionality has to be non-negative.");
	if(mNumComponents < 1)
		throw Exception("The number of components has to be positive.");

	// initialize parameters
	mBiases = -10. * ArrayXd::Random(mNumComponents).abs() - log(mNumComponents);
	mWeights = ArrayXXd::Random(mNumComponents, mNumFeatures).abs() / 100.;
	mFeatures = sampleNormal(mDimInNonlinear, mNumFeatures) / 100.;
	mPredictors = sampleNormal(mNumComponents, mDimInNonlinear) / 100.;
	mLinearPredictor = VectorXd::Zero(mDimInLinear);
}



CMT::STM::STM(int dimInNonlinear, int dimInLinear, int numComponents, int numFeatures) :
	mDimInNonlinear(dimInNonlinear),
	mDimInLinear(dimInLinear),
	mNumComponents(numComponents),
	mNumFeatures(numFeatures < 0 ? dimInNonlinear : numFeatures)
{
	// check hyperparameters
	if(mDimInNonlinear < 0 || mDimInLinear < 0)
		throw Exception("The input dimensionality has to be non-negative.");
	if(mNumComponents < 1)
		throw Exception("The number of components has to be positive.");

	// initialize parameters
	mBiases = -10. * ArrayXd::Random(mNumComponents).abs() - log(mNumComponents);
	mWeights = ArrayXXd::Random(mNumComponents, mNumFeatures).abs() / 100.;
	mFeatures = sampleNormal(mDimInNonlinear, mNumFeatures) / 100.;
	mPredictors = sampleNormal(mNumComponents, mDimInNonlinear) / 100.;
	mLinearPredictor = sampleNormal(mDimInLinear, 1) / 100.;
}



MatrixXd CMT::STM::sample(const MatrixXd& input) const {
	// compute probability of generating a 1
	Array<double, 1, Dynamic> logProb1 =
		logLikelihood(input, MatrixXd::Ones(1, input.cols()));

	// sample uniformly between 0 and 1
	Array<double, 1, Dynamic> uniRand =
		Array<double, 1, Dynamic>::Random(input.cols()).abs();

	return (uniRand < logProb1.exp()).cast<double>();
}



MatrixXd CMT::STM::sample(const MatrixXd& inputNonlinear, const MatrixXd& inputLinear) const {
	// compute probability of generating a 1
	Array<double, 1, Dynamic> logProb1 = logLikelihood(
		inputNonlinear,
		inputLinear,
		MatrixXd::Ones(1, inputNonlinear.cols()));

	// sample uniformly between 0 and 1
	Array<double, 1, Dynamic> uniRand =
		Array<double, 1, Dynamic>::Random(inputNonlinear.cols()).abs();

	return (uniRand < logProb1.exp()).cast<double>();
}



Array<double, 1, Dynamic> CMT::STM::logLikelihood(
	const MatrixXd& input,
	const MatrixXd& output) const
{
	if(input.rows() != dimIn())
		throw Exception("Input has wrong dimensionality.");

	if(dimInLinear())
		// split nonlinear and linear inputs
		return logLikelihood(
			input.topRows(dimInNonlinear()),
			input.bottomRows(dimInLinear()),
			output);

	if(output.rows() != dimOut())
		throw Exception("Output has wrong dimensionality.");
	if(input.cols() != output.cols())
		throw Exception("The number of inputs and outputs must be the same.");

	if(!dimIn()) {
		// model has no inputs and reduces to Bernoulli
		double response = numComponents() > 0 ? log(mBiases.array().exp().sum()) : mBiases[0];
		double logProb1 = -log(1. + exp(-response));
		double logProb0 = -log(1. + exp(response));

		return output.array() * logProb1 + (1. - output.array()) * logProb0;
	}

	// model has only nonlinear inputs
	MatrixXd jointEnergy = mWeights * (mFeatures.transpose() * input).array().square().matrix();
	jointEnergy += mPredictors * input;
	jointEnergy.colwise() += mBiases;

	Array<double, 1, Dynamic> response = logSumExp(jointEnergy);

	// apply logistic sigmoid function
	Array<double, 1, Dynamic> logProb1 = -(1. + (-response).exp()).log();
	Array<double, 1, Dynamic> logProb0 = -(1. + response.exp()).log();

	return output.array() * logProb1 + (1. - output.array()) * logProb0;
}



Array<double, 1, Dynamic> CMT::STM::logLikelihood(
	const MatrixXd& inputNonlinear,
	const MatrixXd& inputLinear,
	const MatrixXd& output) const
{
	if(!dimInLinear())
		return logLikelihood(inputNonlinear, output);

	if(output.rows() != dimOut())
		throw Exception("Output has wrong dimensionality.");
	if(inputNonlinear.rows() != dimInNonlinear() || inputLinear.rows() != dimInLinear())
		throw Exception("Input has wrong dimensionality.");

	Array<double, 1, Dynamic> response;

	if(dimInNonlinear()) {
		// model has linear and nonlinear inputs
		MatrixXd jointEnergy = mWeights * (mFeatures.transpose() * inputNonlinear).array().square().matrix();
		jointEnergy += mPredictors * inputNonlinear;
		jointEnergy.colwise() += mBiases;

		response = logSumExp(jointEnergy) + (mLinearPredictor.transpose() * inputLinear).array();
	} else {
		// model has only linear inputs
		double bias = numComponents() > 0 ? log(mBiases.array().exp().sum()) : mBiases[0];
		response = (mLinearPredictor.transpose() * inputLinear).array() + bias;
	}

	// apply logistic sigmoid function
	Array<double, 1, Dynamic> logProb1 = -(1. + (-response).exp()).log();
	Array<double, 1, Dynamic> logProb0 = -(1. + response.exp()).log();

	return output.array() * logProb1 + (1. - output.array()) * logProb0;
}



void CMT::STM::initialize(const MatrixXd& input, const MatrixXd& output) {
	if(input.rows() != dimIn() || output.rows() != dimOut())
		throw Exception("Data has wrong dimensionality.");
	if(input.cols() != output.cols())
		throw Exception("The number of inputs and outputs should be the same.");

	Array<bool, 1, Dynamic> spikes = output.array() > 0.5;
	int numSpikes = output.sum();

	if(dimInNonlinear() > 0) {
		MatrixXd inputNonlinear = input.topRows(dimInNonlinear());

		if(numSpikes > dimInNonlinear()) {
			MatrixXd inputs1(inputNonlinear.rows(), numSpikes);
			MatrixXd inputs0(inputNonlinear.rows(), spikes.size() - numSpikes);

			// separate data into spike-triggered and non-spike-triggered stimuli
			for(int i = 0, i0 = 0, i1 = 0; i < spikes.size(); ++i)
				if(spikes[i])
					inputs1.col(i1++) = inputNonlinear.col(i);
				else
					inputs0.col(i0++) = inputNonlinear.col(i);

			// spike-triggered/non-spike-triggered mean and precision
			VectorXd m1 = inputs1.rowwise().mean();
			VectorXd m0 = inputs0.rowwise().mean();

			MatrixXd S1 = covariance(inputs1).inverse();
			MatrixXd S0 = covariance(inputs0).inverse();

			// parameters of a quadratic model
			MatrixXd K = (S0 - S1) / 2.;
			VectorXd w = S1 * m1 - S0 * m0;
			double p = static_cast<float>(numSpikes) / output.cols();
			double a = 0.5 * (m0.transpose() * S0 * m0)(0, 0) - 0.5 * (m1.transpose() * S1 * m1)(0, 0)
				+ 0.5 * logDetPD(S1) - 0.5 * logDetPD(S0) + log(p) - log(1. - p)
				- log(mNumComponents);

			// decompose matrix into eigenvectors
			SelfAdjointEigenSolver<MatrixXd> eigenSolver(K);
			VectorXd eigVals = eigenSolver.eigenvalues();
			MatrixXd eigVecs = eigenSolver.eigenvectors();
			VectorXi indices = argSort(eigVals.array().abs());

			// use most informative eigenvectors as features
			for(int i = 0; i < mNumFeatures && i < indices.size(); ++i) {
				int j = indices[i];
				mWeights.col(i).setConstant(eigVals[j]);
				mFeatures.col(i) = eigVecs.col(j);
			}

			mWeights += MatrixXd::Random(mNumComponents, mNumFeatures) / 10.;
			mPredictors.rowwise() = w.transpose();
			mPredictors += sampleNormal(mNumComponents, mDimInNonlinear).matrix() / 1000.;
			mBiases.setConstant(a);
			mBiases += VectorXd::Random(mNumComponents) / 10.;
		}
	}

	if(dimInLinear() > 0) {
		mLinearPredictor = input.bottomRows(dimInLinear()) * output.transpose()
			/ static_cast<float>(numSpikes);
	}
}



bool CMT::STM::train(
	const MatrixXd& inputNonlinear,
	const MatrixXd& inputLinear,
	const MatrixXd& output,
	const Parameters& params)
{
	if(inputNonlinear.cols() != inputLinear.cols())
		throw Exception("Number of nonlinear and linear inputs must be the same.");

	// stack inputs
	MatrixXd input(
		inputNonlinear.rows() + inputLinear.rows(),
		inputLinear.cols());
	input << inputNonlinear, inputLinear;

	return train(input, output, params);
}



bool CMT::STM::train(
	const MatrixXd& inputNonlinear,
	const MatrixXd& inputLinear,
	const MatrixXd& output,
	const MatrixXd& inputNonlinearVal,
	const MatrixXd& inputLinearVal,
	const MatrixXd& outputVal,
	const Parameters& params)
{
	if(inputNonlinear.cols() != inputLinear.cols())
		throw Exception("Number of nonlinear and linear inputs must be the same.");
	if(inputNonlinearVal.cols() != inputLinearVal.cols())
		throw Exception("Number of nonlinear and linear inputs must be the same.");

	// stack inputs
	MatrixXd input(
		inputNonlinear.rows() + inputLinear.rows(),
		inputLinear.cols());
	input << inputNonlinear, inputLinear;

	MatrixXd inputVal(
		inputNonlinearVal.rows() + inputLinearVal.rows(),
		inputLinearVal.cols());
	inputVal << inputNonlinearVal, inputLinearVal;

	return train(input, output, inputVal, outputVal, params);
}



int CMT::STM::numParameters(const Trainable::Parameters& params_) const {
	const Parameters& params = dynamic_cast<const Parameters&>(params_);

	int numParams = 0;
	if(params.trainBiases)
		numParams += mBiases.size();
	if(params.trainWeights)
		numParams += mWeights.size();
	if(params.trainFeatures)
		numParams += mFeatures.size();
	if(params.trainPredictors)
		numParams += mPredictors.size();
	if(params.trainLinearPredictor)
		numParams += mLinearPredictor.size();
	return numParams;
}



lbfgsfloatval_t* CMT::STM::parameters(const Trainable::Parameters& params_) const {
	const Parameters& params = dynamic_cast<const Parameters&>(params_);

	lbfgsfloatval_t* x = lbfgs_malloc(numParameters(params));

	int k = 0;
	if(params.trainBiases)
		for(int i = 0; i < mBiases.size(); ++i, ++k)
			x[k] = mBiases.data()[i];
	if(params.trainWeights)
		for(int i = 0; i < mWeights.size(); ++i, ++k)
			x[k] = mWeights.data()[i];
	if(params.trainFeatures)
		for(int i = 0; i < mFeatures.size(); ++i, ++k)
			x[k] = mFeatures.data()[i];
	if(params.trainPredictors)
		for(int i = 0; i < mPredictors.size(); ++i, ++k)
			x[k] = mPredictors.data()[i];
	if(params.trainLinearPredictor)
		for(int i = 0; i < mLinearPredictor.size(); ++i, ++k)
			x[k] = mLinearPredictor.data()[i];

	return x;
}



void CMT::STM::setParameters(const lbfgsfloatval_t* x, const Trainable::Parameters& params_) {
	const Parameters& params = dynamic_cast<const Parameters&>(params_);

	int offset = 0;

	if(params.trainBiases) {
		mBiases = VectorLBFGS(const_cast<double*>(x), mNumComponents);
		offset += mBiases.size();
	}

	if(params.trainWeights) {
		mWeights = MatrixLBFGS(const_cast<double*>(x) + offset, mNumComponents, mNumFeatures);
		offset += mWeights.size();
	}

	if(params.trainFeatures) {
		mFeatures = MatrixLBFGS(const_cast<double*>(x) + offset, dimInNonlinear(), mNumFeatures);
		offset += mFeatures.size();
	}

	if(params.trainPredictors) {
		mPredictors = MatrixLBFGS(const_cast<double*>(x) + offset, mNumComponents, dimInNonlinear());
		offset += mPredictors.size();
	}

	if(params.trainLinearPredictor) {
		mLinearPredictor = VectorLBFGS(const_cast<double*>(x) + offset, dimInLinear());
		offset += mLinearPredictor.size();
	}
}



double CMT::STM::parameterGradient(
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

	VectorLBFGS biases(params.trainBiases ? y : const_cast<double*>(mBiases.data()), mNumComponents);
	VectorLBFGS biasesGrad(g, mNumComponents);
	if(params.trainBiases)
		offset += biases.size();

	MatrixLBFGS weights(params.trainWeights ? y + offset :
		const_cast<double*>(mWeights.data()), mNumComponents, mNumFeatures);
	MatrixLBFGS weightsGrad(g + offset, mNumComponents, mNumFeatures);
	if(params.trainWeights)
		offset += weights.size();

	MatrixLBFGS features(params.trainFeatures ? y + offset :
		const_cast<double*>(mFeatures.data()), dimInNonlinear(), mNumFeatures);
	MatrixLBFGS featuresGrad(g + offset, dimInNonlinear(), mNumFeatures);
	if(params.trainFeatures)
		offset += features.size();

	MatrixLBFGS predictors(params.trainPredictors ? y + offset :
		const_cast<double*>(mPredictors.data()), mNumComponents, dimInNonlinear());
	MatrixLBFGS predictorsGrad(g + offset, mNumComponents, dimInNonlinear());
	if(params.trainPredictors)
		offset += predictors.size();

	VectorLBFGS linearPredictor(params.trainLinearPredictor ? y + offset :
		const_cast<double*>(mLinearPredictor.data()), dimInLinear());
	VectorLBFGS linearPredictorGrad(g + offset, dimInLinear());
	if(params.trainLinearPredictor)
		offset += linearPredictor.size();

	if(g) {
		// initialize gradients
		if(params.trainBiases)
			biasesGrad.setZero();
		if(params.trainWeights)
			weightsGrad.setZero();
		if(params.trainFeatures)
			featuresGrad.setZero();
		if(params.trainPredictors)
			predictorsGrad.setZero();
		if(params.trainLinearPredictor)
			linearPredictorGrad.setZero();
	}

	// split data into batches for better performance
	int numData = static_cast<int>(inputCompl.cols());
	int batchSize = min(max(params.batchSize, 10), numData);

	#pragma omp parallel for
	for(int b = 0; b < inputCompl.cols(); b += batchSize) {
		// TODO: can this be done more efficiently?
		int width = min(batchSize, numData - b);
		const MatrixXd& inputNonlinear = inputCompl.block(0, b, dimInNonlinear(), width);
		const MatrixXd& inputLinear = inputCompl.block(dimInNonlinear(), b, dimInLinear(), width);
		const MatrixXd& output = outputCompl.middleCols(b, width);

		ArrayXXd featureOutput = features.transpose() * inputNonlinear;
		MatrixXd featureOutputSq = featureOutput.square();
		MatrixXd weightsOutput = weights * featureOutputSq;
		ArrayXXd predictorOutput = predictors * inputNonlinear;

		MatrixXd jointEnergy = weights * featureOutputSq + predictors * inputNonlinear;
		jointEnergy.colwise() += biases;

		Matrix<double, 1, Dynamic> response = logSumExp(jointEnergy);

		// posterior over components for each data point
		MatrixXd posterior = (jointEnergy.rowwise() - response).array().exp();

		if(dimInLinear())
			response += linearPredictor.transpose() * inputLinear;

		// apply logistic sigmoid function
		Array<double, 1, Dynamic> prob1 = 1. / (1. + (-response.array()).exp());
		Array<double, 1, Dynamic> prob0 = 1. - prob1;

		double logLikBatch = (output.array() * prob1.log() + (1. - output.array()) * prob0.log()).sum();

		#pragma omp critical
		logLik += logLikBatch;

		if(!g)
			// don't compute gradients
			continue;

		Array<double, 1, Dynamic> tmp = output.array() * prob0 - (1. - output.array()) * prob1;

		MatrixXd postTmp = posterior.array().rowwise() * tmp;

		if(params.trainBiases)
			#pragma omp critical
			biasesGrad -= postTmp.rowwise().sum();

		if(params.trainWeights)
			#pragma omp critical
			weightsGrad -= postTmp * featureOutputSq.transpose();

		if(params.trainFeatures) {
			ArrayXXd tmp2 = 2. * weights.transpose() * postTmp;
			MatrixXd tmp3 = featureOutput * tmp2;
			#pragma omp critical
			featuresGrad -= inputNonlinear * tmp3.transpose();
		}

		if(params.trainPredictors)
			#pragma omp critical
			predictorsGrad -= postTmp * inputNonlinear.transpose();

		if(params.trainLinearPredictor && dimInLinear())
			#pragma omp critical
			linearPredictorGrad -= inputLinear * tmp.transpose().matrix();
	}

	double normConst = inputCompl.cols() * log(2.) * dimOut();

	if(g) {
		for(int i = 0; i < offset; ++i)
			g[i] /= normConst;

		switch(params.regularizer) {
			case Parameters::L1:
				if(params.trainFeatures && params.regularizeFeatures > 0.)
					featuresGrad += params.regularizeFeatures * signum(features);

				if(params.trainPredictors && params.regularizePredictors > 0.)
					predictorsGrad += params.regularizePredictors * signum(predictors);

				if(params.trainWeights && params.regularizeWeights > 0.)
					weightsGrad += params.regularizeWeights * signum(weights);

				if(params.trainLinearPredictor && params.regularizeLinearPredictor > 0.)
					linearPredictorGrad += params.regularizeLinearPredictor * signum(linearPredictor);

				break;

			case Parameters::L2:
				if(params.trainFeatures && params.regularizeFeatures > 0.)
					featuresGrad += params.regularizeFeatures * 2. * features;

				if(params.trainPredictors && params.regularizePredictors > 0.)
					predictorsGrad += params.regularizePredictors * 2. * predictors;

				if(params.trainWeights && params.regularizeWeights > 0.)
					weightsGrad += params.regularizeWeights * 2. * weights;

				if(params.trainLinearPredictor && params.regularizeLinearPredictor > 0.)
					linearPredictorGrad += params.regularizeLinearPredictor * 2. * linearPredictor;

				break;
		}
	}

	double value = -logLik / normConst;

	switch(params.regularizer) {
		case Parameters::L1:
			if(params.trainFeatures && params.regularizeFeatures > 0.)
				value += params.regularizeFeatures * features.array().abs().sum();

			if(params.trainPredictors && params.regularizePredictors > 0.)
				value += params.regularizePredictors * predictors.array().abs().sum();

			if(params.trainWeights && params.regularizeWeights > 0.)
				value += params.regularizeWeights * weights.array().abs().sum();

			if(params.trainLinearPredictor && params.regularizeLinearPredictor > 0.)
				value += params.regularizeLinearPredictor * linearPredictor.array().abs().sum();

			break;

		case Parameters::L2:
			if(params.trainFeatures && params.regularizeFeatures > 0.)
				value += params.regularizeFeatures * features.array().square().sum();

			if(params.trainPredictors && params.regularizePredictors > 0.)
				value += params.regularizePredictors * predictors.array().square().sum();

			if(params.trainWeights && params.regularizeWeights > 0.)
				value += params.regularizeWeights * weights.array().square().sum();

			if(params.trainLinearPredictor && params.regularizeLinearPredictor > 0.)
				value += params.regularizeLinearPredictor * linearPredictor.array().square().sum();
			
			break;
	}

	return value;
}



pair<pair<ArrayXXd, ArrayXXd>, Array<double, 1, Dynamic> > CMT::STM::computeDataGradient(
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



bool CMT::STM::train(
	const MatrixXd& input,
	const MatrixXd& output,
	const MatrixXd* inputVal,
	const MatrixXd* outputVal,
	const Trainable::Parameters& params)
{
	if(!dimIn()) {
		// STM reduces to Bernoulli
		double prob = output.array().mean();
		mBiases.setConstant(prob > 0. ? log(prob) - log(1. - prob) - log(numComponents()) : -50.);
		return true;
	} else if(!dimInNonlinear()) {
		if(dimInLinear() != input.rows())
			throw Exception("Input has wrong dimensionality.");

		// STM reduces to GLM
		LogisticFunction nonlinearity;
		Bernoulli distribution;
		GLM glm(dimInLinear(), &nonlinearity, &distribution);

		bool converged;
		if(inputVal && outputVal)
			converged = glm.train(input, output, *inputVal, *outputVal, params);
		else
			converged = glm.train(input, output, params);

		// copy parameters
		mLinearPredictor = glm.weights();
		mBiases.setConstant(glm.bias() - log(numComponents()));

		return converged;
	} else {
		return Trainable::train(input, output, inputVal, outputVal, params);
	}
}
