#include "glm.h"
#include "utils.h"
#include "Eigen/Cholesky"
#include "Eigen/LU"

#include "stm.h"
using CMT::STM;

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

#include "nonlinearities.h"
using CMT::Nonlinearity;
using CMT::LogisticFunction;

#include "univariatedistributions.h"
using CMT::UnivariateDistribution;
using CMT::Bernoulli;

#include <utility>
using std::pair;
using std::make_pair;

#include <cmath>
using std::max;
using std::min;

#include <limits>
using std::numeric_limits;

Nonlinearity* const STM::defaultNonlinearity = new LogisticFunction;
UnivariateDistribution* const STM::defaultDistribution = new Bernoulli;

CMT::STM::Parameters::Parameters() :
	Trainable::Parameters::Parameters(),
	trainSharpness(false),
	trainBiases(true),
	trainWeights(true),
	trainFeatures(true),
	trainPredictors(true),
	trainLinearPredictor(true),
	regularizeBiases(0.),
	regularizeWeights(0.),
	regularizeFeatures(0.),
	regularizePredictors(0.),
	regularizeLinearPredictor(0.)
{
}



CMT::STM::Parameters::Parameters(const Parameters& params) :
	Trainable::Parameters::Parameters(params),
	trainSharpness(params.trainSharpness),
	trainBiases(params.trainBiases),
	trainWeights(params.trainWeights),
	trainFeatures(params.trainFeatures),
	trainPredictors(params.trainPredictors),
	trainLinearPredictor(params.trainLinearPredictor),
	regularizeBiases(params.regularizeBiases),
	regularizeWeights(params.regularizeWeights),
	regularizeFeatures(params.regularizeFeatures),
	regularizePredictors(params.regularizePredictors),
	regularizeLinearPredictor(params.regularizeLinearPredictor)
{
}



CMT::STM::Parameters& CMT::STM::Parameters::operator=(const Parameters& params) {
	Trainable::Parameters::operator=(params);

	trainSharpness = params.trainSharpness;
	trainBiases = params.trainBiases;
	trainWeights = params.trainWeights;
	trainFeatures = params.trainFeatures;
	trainPredictors = params.trainPredictors;
	trainLinearPredictor = params.trainLinearPredictor;
	regularizeBiases = params.regularizeBiases;
	regularizeWeights = params.regularizeWeights;
	regularizeFeatures = params.regularizeFeatures;
	regularizePredictors = params.regularizePredictors;
	regularizeLinearPredictor = params.regularizeLinearPredictor;

	return *this;
}



CMT::STM::STM(
	int dimIn,
	int numComponents,
	int numFeatures,
	Nonlinearity* nonlinearity,
	UnivariateDistribution* distribution) :
	mDimInNonlinear(dimIn),
	mDimInLinear(0),
	mNumComponents(numComponents),
	mNumFeatures(numFeatures < 0 ? dimIn : numFeatures),
	mNonlinearity(nonlinearity ? nonlinearity : defaultNonlinearity),
	mDistribution(distribution ? distribution : defaultDistribution)
{
	// check hyperparameters
	if(mDimInNonlinear < 0)
		throw Exception("The input dimensionality has to be non-negative.");
	if(mNumComponents < 1)
		throw Exception("The number of components has to be positive.");

	// initialize parameters
	mSharpness = 1.;
	mBiases = -10. * ArrayXd::Random(mNumComponents).abs() - log(mNumComponents);
	mWeights = ArrayXXd::Random(mNumComponents, mNumFeatures).abs() / 100.;
	mFeatures = sampleNormal(mDimInNonlinear, mNumFeatures) / 100.;
	mPredictors = sampleNormal(mNumComponents, mDimInNonlinear) / 100.;
	mLinearPredictor = VectorXd::Zero(mDimInLinear);
}



CMT::STM::STM(
	int dimInNonlinear,
	int dimInLinear,
	int numComponents,
	int numFeatures,
	Nonlinearity* nonlinearity,
	UnivariateDistribution* distribution) :
	mDimInNonlinear(dimInNonlinear),
	mDimInLinear(dimInLinear),
	mNumComponents(numComponents),
	mNumFeatures(numFeatures < 0 ? dimInNonlinear : numFeatures),
	mNonlinearity(nonlinearity ? nonlinearity : defaultNonlinearity),
	mDistribution(distribution ? distribution : defaultDistribution)
{
	// check hyperparameters
	if(mDimInNonlinear < 0 || mDimInLinear < 0)
		throw Exception("The input dimensionality has to be non-negative.");
	if(mNumComponents < 1)
		throw Exception("The number of components has to be positive.");

	// initialize parameters
	mSharpness = 1.;
	mBiases = -10. * ArrayXd::Random(mNumComponents).abs() - log(mNumComponents);
	mWeights = ArrayXXd::Random(mNumComponents, mNumFeatures).abs() / 100.;
	mFeatures = sampleNormal(mDimInNonlinear, mNumFeatures) / 100.;
	mPredictors = sampleNormal(mNumComponents, mDimInNonlinear) / 100.;
	mLinearPredictor = sampleNormal(mDimInLinear, 1) / 100.;
}



MatrixXd CMT::STM::sample(const MatrixXd& input) const {
	return mDistribution->sample(mNonlinearity->operator()(response(input)));
}



MatrixXd CMT::STM::sample(const MatrixXd& inputNonlinear, const MatrixXd& inputLinear) const {
	return mDistribution->sample(
		mNonlinearity->operator()(response(inputNonlinear, inputLinear)));
}



MatrixXd CMT::STM::predict(const MatrixXd& input) const {
	return mNonlinearity->operator()(response(input));
}



MatrixXd CMT::STM::predict(const MatrixXd& inputNonlinear, const MatrixXd& inputLinear) const {
	return mNonlinearity->operator()(response(inputNonlinear, inputLinear));
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

	return mDistribution->logLikelihood(
		output,
		mNonlinearity->operator()(response(input)));
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

	return mDistribution->logLikelihood(
		output,
		mNonlinearity->operator()(response(inputNonlinear, inputLinear)));
}



Array<double, 1, Dynamic> CMT::STM::response(const MatrixXd& input) const {
	if(input.rows() != dimIn())
		throw Exception("Input has wrong dimensionality.");

	if(dimInLinear())
		// split nonlinear and linear inputs
		return response(
			input.topRows(dimInNonlinear()),
			input.bottomRows(dimInLinear()));

	Array<double, 1, Dynamic> response;

	if(!dimIn())
		// model has no inputs and reduces to univariate distribution
		return Array<double, 1, Dynamic>::Constant(input.cols(),
			numComponents() > 1 ? log(mBiases.array().exp().sum()) : mBiases[0]);

	// model has only nonlinear inputs
	MatrixXd jointEnergy;
	if(numFeatures() > 0)
		jointEnergy = mWeights * (mFeatures.transpose() * input).array().square().matrix()
			+ mPredictors * input;
	else
		jointEnergy = mPredictors * input;
	jointEnergy.colwise() += mBiases;

	return logSumExp(mSharpness * jointEnergy) / mSharpness;
}



Array<double, 1, Dynamic> CMT::STM::response(
	const MatrixXd& inputNonlinear,
	const MatrixXd& inputLinear) const
{
	if(!dimInLinear())
		return response(inputNonlinear);

	if(inputNonlinear.rows() != dimInNonlinear() || inputLinear.rows() != dimInLinear())
		throw Exception("Input has wrong dimensionality.");

	Array<double, 1, Dynamic> response;

	if(!dimInNonlinear()) {
		// model has only linear inputs
		double bias = numComponents() > 1 ?
			log((mSharpness * mBiases).array().exp().sum()) / mSharpness :
			mBiases[0];
		return (mLinearPredictor.transpose() * inputLinear).array() + bias;
	}

	// model has linear and nonlinear inputs
	MatrixXd jointEnergy;
	if(numFeatures() > 0)
		jointEnergy = mWeights * (mFeatures.transpose() * inputNonlinear).array().square().matrix()
			+ mPredictors * inputNonlinear;
	else
		jointEnergy = mPredictors * inputNonlinear;
	jointEnergy.colwise() += mBiases;

	return logSumExp(mSharpness * jointEnergy) / mSharpness
		+ (mLinearPredictor.transpose() * inputLinear).array();
}



ArrayXXd CMT::STM::nonlinearResponses(const MatrixXd& input) const {
	if(input.rows() != dimInNonlinear() && input.rows() != dimIn())
		throw Exception("Input has wrong dimensionality.");

	if(!dimInNonlinear())
		// model has only linear inputs
		return MatrixXd::Zero(numComponents(), input.cols()).colwise() + mBiases;

	MatrixXd jointEnergy;
	if(numFeatures() > 0)
		jointEnergy = mWeights * (mFeatures.transpose() * input.topRows(dimInNonlinear())).array().square().matrix()
			+ mPredictors * input.topRows(dimInNonlinear());
	else
		jointEnergy = mPredictors * input.topRows(dimInNonlinear());
	jointEnergy.colwise() += mBiases;

	return jointEnergy;
}



ArrayXXd CMT::STM::linearResponse(const MatrixXd& input) const {
	if(input.rows() != dimInLinear() && input.rows() != dimIn())
		throw Exception("Input has wrong dimensionality.");

	if(!dimInLinear())
		// model has only linear inputs
		return MatrixXd::Zero(1, input.cols());

	return mLinearPredictor.transpose() * input.bottomRows(dimInLinear());
}



void CMT::STM::initialize(const MatrixXd& input, const MatrixXd& output) {
	if(input.rows() != dimIn() || output.rows() != dimOut())
		throw Exception("Data has wrong dimensionality.");
	if(input.cols() != output.cols())
		throw Exception("The number of inputs and outputs should be the same.");

	Array<bool, 1, Dynamic> spikes = output.array() > 0.5;
	int numSpikes = spikes.sum();

	if(numSpikes > dimInNonlinear() && dimInNonlinear() > 0) {
		mSharpness = 1.;
		MatrixXd inputNonlinear = input.topRows(dimInNonlinear());
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

		mWeights = mWeights.array() * (0.5 + 0.5 * ArrayXXd::Random(mNumComponents, mNumFeatures).abs());
		mPredictors.rowwise() = w.transpose();
		mPredictors += sampleNormal(mNumComponents, mDimInNonlinear).matrix() * log(mNumComponents) / 10.;
		mBiases.setConstant(a);
		mBiases += VectorXd::Random(mNumComponents) * log(mNumComponents) / 100.;
	}

	if(dimInLinear() > 0)
		mLinearPredictor = input.bottomRows(dimInLinear()) * output.transpose() / numSpikes;
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
	if(params.trainSharpness)
		numParams += 1;
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
	if(params.trainSharpness)
		x[k++] = mSharpness;

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

	if(params.trainSharpness)
		mSharpness = x[offset++];
}



double CMT::STM::parameterGradient(
	const MatrixXd& inputCompl,
	const MatrixXd& outputCompl,
	const lbfgsfloatval_t* x,
	lbfgsfloatval_t* g,
	const Trainable::Parameters& params_) const
{
 	// check if nonlinearity is differentiable
 	DifferentiableNonlinearity* nonlinearity = dynamic_cast<DifferentiableNonlinearity*>(mNonlinearity);

	if(!nonlinearity)
		throw Exception("Nonlinearity has to be differentiable for training.");

	const Parameters& params = dynamic_cast<const Parameters&>(params_);

	// average log-likelihood
	double logLik = 0.;

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

	double sharpness = params.trainSharpness ? y[offset++] : mSharpness;
	double sharpnessGrad = 0.;

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
		int width = min(batchSize, numData - b);
		const MatrixXd& inputNonlinear = inputCompl.block(0, b, dimInNonlinear(), width);
		const MatrixXd& inputLinear = inputCompl.block(dimInNonlinear(), b, dimInLinear(), width);
		const MatrixXd& output = outputCompl.middleCols(b, width);

		ArrayXXd featureOutput;
		MatrixXd featureOutputSq;
		MatrixXd jointEnergy;

		if(numFeatures() > 0) {
			featureOutput = features.transpose() * inputNonlinear;
			featureOutputSq = featureOutput.square();
			jointEnergy = weights * featureOutputSq + predictors * inputNonlinear;
		} else {
			jointEnergy = predictors * inputNonlinear;
		}

		jointEnergy.colwise() += biases;
		MatrixXd jointEnergyScaled = jointEnergy * sharpness;

		Matrix<double, 1, Dynamic> response = logSumExp(jointEnergyScaled);

		// posterior over components for each data point
		MatrixXd posterior = (jointEnergyScaled.rowwise() - response).array().exp();

		response /= sharpness;

		MatrixXd nonlinearResponse;
		if(params.trainSharpness)
			// make copy of nonlinear response
			nonlinearResponse = response;

		if(dimInLinear())
			response += linearPredictor.transpose() * inputLinear;

		// update log-likelihood
		double logLikBatch = mDistribution->logLikelihood(
			output,
			nonlinearity->operator()(response)).sum();

		#pragma omp critical
		logLik += logLikBatch;

		if(!g)
			// don't compute gradients
			continue;

		Array<double, 1, Dynamic> tmp = -mDistribution->gradient(output, nonlinearity->operator()(response))
			* nonlinearity->derivative(response);

		MatrixXd postTmp = posterior.array().rowwise() * tmp;

		if(params.trainBiases)
			#pragma omp critical
			biasesGrad -= postTmp.rowwise().sum();

		if(numFeatures() > 0) {
			if(params.trainWeights)
				#pragma omp critical
				weightsGrad -= postTmp * featureOutputSq.transpose();

			if(params.trainFeatures) {
				ArrayXXd tmp2 = 2. * weights.transpose() * postTmp;
				MatrixXd tmp3 = featureOutput * tmp2;
				#pragma omp critical
				featuresGrad -= inputNonlinear * tmp3.transpose();
			}
		}

		if(params.trainPredictors)
			#pragma omp critical
			predictorsGrad -= postTmp * inputNonlinear.transpose();

		if(params.trainLinearPredictor && dimInLinear() > 0)
			#pragma omp critical
			linearPredictorGrad -= inputLinear * tmp.transpose().matrix();

		if(params.trainSharpness) {
			double tmp2 = ((jointEnergy.array() * posterior.array()).colwise().sum() * tmp).sum() / sharpness;
			double tmp3 = (nonlinearResponse.array() * tmp).sum() / sharpness;
			#pragma omp critical
			sharpnessGrad -= tmp2 - tmp3;
		}
	}

	double normConst = inputCompl.cols() * log(2.) * dimOut();

	if(g) {
		if(params.trainSharpness)
			g[offset - 1] = sharpnessGrad;

		for(int i = 0; i < offset; ++i)
			g[i] /= normConst;

		if(params.trainBiases)
			biasesGrad += params.regularizeBiases.gradient(biases);
		if(params.trainFeatures)
			featuresGrad += params.regularizeFeatures.gradient(features);
		if(params.trainPredictors)
			predictorsGrad += params.regularizePredictors.gradient(predictors.transpose()).transpose();
		if(params.trainWeights)
			weightsGrad += params.regularizeWeights.gradient(weights.transpose()).transpose();
		if(params.trainLinearPredictor)
			linearPredictorGrad += params.regularizeLinearPredictor.gradient(linearPredictor);
	}

	double value = -logLik / normConst;

	value += params.regularizeBiases.evaluate(biases);
	value += params.regularizeFeatures.evaluate(features);
	value += params.regularizePredictors.evaluate(predictors.transpose());
	value += params.regularizeWeights.evaluate(weights.transpose());
	value += params.regularizeLinearPredictor.evaluate(linearPredictor);

	if(value != value)
		// value is NaN; line search probably went into bad region of parameter space
		value = numeric_limits<double>::max();

	return value;
}



pair<pair<ArrayXXd, ArrayXXd>, Array<double, 1, Dynamic> > CMT::STM::computeDataGradient(
	const MatrixXd& input,
	const MatrixXd& output) const
{
	// make sure nonlinearity is differentiable
	DifferentiableNonlinearity* nonlinearity =
		dynamic_cast<DifferentiableNonlinearity*>(mNonlinearity);
	if(!nonlinearity)
		throw Exception("Nonlinearity has to be differentiable.");

	if(input.rows() != dimIn())
		throw Exception("Input has wrong dimensionality.");
	if(output.rows() != 1)
		throw Exception("Output has wrong dimensionality.");
	if(input.cols() != output.cols())
		throw Exception("Number of inputs and outputs should be the same.");

	if(dimInNonlinear() && !dimInLinear()) {
		Array<double, 1, Dynamic> responses;

		ArrayXXd jointEnergy;

		if(numFeatures() > 0)
			jointEnergy = mWeights * (mFeatures.transpose() * input).array().square().matrix()
				+ mPredictors * input;
		else
			jointEnergy = mPredictors * input;
		jointEnergy.colwise() += mBiases.array();
		jointEnergy *= mSharpness;

		responses = logSumExp(jointEnergy);

		// posterior over components for each input
		MatrixXd posterior = (jointEnergy.rowwise() - responses).array().exp();

		responses /= mSharpness;

		Array<double, 1, Dynamic> tmp0 = (*mNonlinearity)(responses);
		Array<double, 1, Dynamic> tmp1 = -mDistribution->gradient(output, tmp0);
		Array<double, 1, Dynamic> tmp2 = nonlinearity->derivative(responses);

		ArrayXXd avgPredictor = mPredictors.transpose() * posterior;

		ArrayXXd tmp3;
		if(numFeatures() > 0) {
			ArrayXXd avgWeights = (2. * mWeights).transpose() * posterior;
			tmp3 = mFeatures * (avgWeights * (mFeatures.transpose() * input).array()).matrix();
		} else {
			tmp3 = ArrayXXd::Zero(avgPredictor.rows(), avgPredictor.cols());
		}

		return make_pair(
			make_pair(
				(tmp3 + avgPredictor).rowwise() * (tmp1 * tmp2),
				ArrayXXd::Zero(output.rows(), output.cols())),
			mDistribution->logLikelihood(output, tmp0));

	} else if(dimInNonlinear() && dimInLinear()) {
		// split inputs into linear and nonlinear components
		MatrixXd inputNonlinear = input.topRows(dimInNonlinear());
		MatrixXd inputLinear = input.bottomRows(dimInLinear());

		Array<double, 1, Dynamic> responses;

		ArrayXXd jointEnergy;

		if(numFeatures() > 0)
			jointEnergy = mWeights * (mFeatures.transpose() * inputNonlinear).array().square().matrix()
				+ mPredictors * input;
		else
			jointEnergy = mPredictors * inputNonlinear;
		jointEnergy.colwise() += mBiases.array();
		jointEnergy *= mSharpness;

		responses = logSumExp(jointEnergy);

		// posterior over components for each input
		MatrixXd posterior = (jointEnergy.rowwise() - responses).array().exp();

		responses /= mSharpness;
		responses += (mLinearPredictor.transpose() * inputLinear).array();

		Array<double, 1, Dynamic> tmp0 = (*mNonlinearity)(responses);
		Array<double, 1, Dynamic> tmp1 = -mDistribution->gradient(output, tmp0);
		Array<double, 1, Dynamic> tmp2 = nonlinearity->derivative(responses);

		ArrayXXd avgPredictor = mPredictors.transpose() * posterior;

		ArrayXXd tmp3;
		if(numFeatures() > 0) {
			ArrayXXd avgWeights = (2. * mWeights).transpose() * posterior;
			tmp3 = mFeatures * (avgWeights * (mFeatures.transpose() * inputNonlinear).array()).matrix();
		} else {
			tmp3 = ArrayXXd::Zero(avgPredictor.rows(), avgPredictor.cols());
		}

		// concatenate gradients of nonlinear and linear component
		ArrayXXd inputGradient(dimIn(), input.cols());
		inputGradient << 
			(tmp3 + avgPredictor).rowwise() * (tmp1 * tmp2),
			mLinearPredictor * (tmp1 * tmp2).matrix();

		return make_pair(
			make_pair(
				inputGradient,
				ArrayXXd::Zero(output.rows(), output.cols())),
			mDistribution->logLikelihood(output, tmp0));

	} else if(dimInLinear()) {
		double avgBias = logSumExp(mSharpness * mBiases)(0, 0) / mSharpness;
		Array<double, 1, Dynamic> responses = (mLinearPredictor.transpose() * input).array() + avgBias;

		Array<double, 1, Dynamic> tmp0 = (*mNonlinearity)(responses);
		Array<double, 1, Dynamic> tmp1 = -mDistribution->gradient(output, tmp0);
		Array<double, 1, Dynamic> tmp2 = nonlinearity->derivative(responses);

		return make_pair(
			make_pair(
				mLinearPredictor * (tmp1 * tmp2).matrix(),
				ArrayXXd::Zero(output.rows(), output.cols())),
			mDistribution->logLikelihood(output, tmp0));
	}

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
		// STM reduces to univariate distribution
		InvertibleNonlinearity* nonlinearity = dynamic_cast<InvertibleNonlinearity*>(mNonlinearity);

		if(!nonlinearity)
			throw Exception("Nonlinearity has to be invertible for training.");

		double mean = output.array().mean();
		if(mean >= 0. && mean < 1e-50)
			mean = 1e-50;

		mBiases.setConstant(nonlinearity->inverse(mean) - log(numComponents()));

		return true;

	} else if(!dimInNonlinear() || (numComponents() == 1 && numFeatures() == 0)) {
		if(dimIn() != input.rows())
			throw Exception("Input has wrong dimensionality.");

		// STM reduces to GLM
		GLM glm(dimIn(), mNonlinearity, mDistribution);

		GLM::Parameters glmParams;
		glmParams.Trainable::Parameters::operator=(params);

		const Parameters& stmParams = dynamic_cast<const Parameters&>(params);

		if(stmParams.trainLinearPredictor)
			glmParams.trainWeights = true;
		if(stmParams.trainBiases)
			glmParams.trainBias = true;
		glmParams.regularizeWeights = stmParams.regularizeLinearPredictor;
		glmParams.regularizeBias = stmParams.regularizeBiases;

		bool converged;
		if(inputVal && outputVal)
			converged = glm.train(input, output, *inputVal, *outputVal, glmParams);
		else
			converged = glm.train(input, output, glmParams);

		// copy parameters
		mPredictors = glm.weights().topRows(dimInNonlinear()).transpose();
		mLinearPredictor = glm.weights().bottomRows(dimInLinear());
		mBiases.setConstant(glm.bias() - log(numComponents()));

		return converged;

	} else {
		return Trainable::train(input, output, inputVal, outputVal, params);
	}
}
