#include "mcgsm.h"
#include "utils.h"
#include "Eigen/Eigenvalues"
#include <sys/time.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <utility>

using namespace std;

typedef pair<pair<const MCGSM*, const MCGSM::Parameters*>, pair<const MatrixXd*, const MatrixXd*> > ParamsLBFGS;
typedef Map<Matrix<lbfgsfloatval_t, Dynamic, Dynamic> > MatrixLBFGS;

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
	const MCGSM& mcgsm = *static_cast<ParamsLBFGS*>(instance)->first.first;
	const MCGSM::Parameters& params = *static_cast<ParamsLBFGS*>(instance)->first.second;

	if(params.verbosity > 0)
		cout << setw(6) << iteration << setw(10) << setprecision(5) << fx << endl;

	if(params.callback && iteration % params.cbIter == 0) {
		// TODO: fix this nasty hack
		const_cast<MCGSM&>(mcgsm).setParameters(x, params);

		if(!(*params.callback)(iteration, mcgsm))
			return 1;
	}


	return 0;
}



lbfgsfloatval_t evaluateLBFGS(
	void* instance, 
	const lbfgsfloatval_t* x, 
	lbfgsfloatval_t* g, 
	int, double) 
{
	// unpack user data
	const MCGSM& mcgsm = *static_cast<ParamsLBFGS*>(instance)->first.first;
	const MCGSM::Parameters& params = *static_cast<ParamsLBFGS*>(instance)->first.second;
	const MatrixXd& input = *static_cast<ParamsLBFGS*>(instance)->second.first;
	const MatrixXd& output = *static_cast<ParamsLBFGS*>(instance)->second.second;

	return mcgsm.computeGradient(input, output, x, g, params);
}



MCGSM::Callback::~Callback() {
}



MCGSM::Parameters::Parameters() {
	verbosity = 0;
	maxIter = 1000;
	threshold = 1e-5;
	numGrad = 20;
	batchSize = 2000;
	callback = 0;
	cbIter = 25;
	trainPriors = true;
	trainScales = true;
	trainWeights = true;
	trainFeatures = true;
	trainCholeskyFactors = true;
	trainPredictors = true;
}



MCGSM::Parameters::Parameters(const Parameters& params) :
	verbosity(params.verbosity),
	maxIter(params.maxIter),
	threshold(params.threshold),
	numGrad(params.numGrad),
	batchSize(params.batchSize),
	callback(0),
	cbIter(params.cbIter),
	trainPriors(params.trainPriors),
	trainScales(params.trainScales),
	trainWeights(params.trainWeights),
	trainFeatures(params.trainFeatures),
	trainCholeskyFactors(params.trainCholeskyFactors),
	trainPredictors(params.trainPredictors)
{
	if(params.callback)
		callback = params.callback->copy();
}



MCGSM::Parameters::~Parameters() {
	if(callback)
		delete callback;
}



MCGSM::Parameters& MCGSM::Parameters::operator=(const Parameters& params) {
	verbosity = params.verbosity;
	maxIter = params.maxIter;
	threshold = params.threshold;
	numGrad = params.numGrad;
	batchSize = params.batchSize;
	callback = params.callback ? params.callback->copy() : 0;
	cbIter = params.cbIter;
	trainPriors = params.trainPriors;
	trainScales = params.trainScales;
	trainWeights = params.trainWeights;
	trainFeatures = params.trainFeatures;
	trainCholeskyFactors = params.trainCholeskyFactors;
	trainPredictors = params.trainPredictors;

	return *this;
}



MCGSM::MCGSM(
	int dimIn, 
	int dimOut,
	int numComponents,
	int numScales,
	int numFeatures) :
	mDimIn(dimIn),
	mDimOut(dimOut),
	mNumComponents(numComponents),
	mNumScales(numScales),
	mNumFeatures(numFeatures < 0 ? mDimIn : numFeatures)
{
	// check hyperparameters
	if(mDimOut < 1)
		throw Exception("The number of output dimensions has to be positive.");
	if(mNumScales < 1)
		throw Exception("The number of scales has to be positive.");
	if(mNumComponents < 1)
		throw Exception("The number of components has to be positive.");
	if(mNumFeatures < 1)
		throw Exception("The number of features has to be positive.");

	// initialize parameters
	mPriors = ArrayXXd::Zero(mNumComponents, mNumScales);
	mScales = ArrayXXd::Random(mNumComponents, mNumScales);
	mWeights = ArrayXXd::Random(mNumComponents, mNumFeatures).abs() / 100. + 0.01;
	mFeatures = sampleNormal(mDimIn, mNumFeatures) / 100.;

	for(int i = 0; i < mNumComponents; ++i) {
		mCholeskyFactors.push_back(MatrixXd::Identity(mDimOut, mDimOut));
		mPredictors.push_back(sampleNormal(mDimOut, mDimIn) / 10.);
	}
}



MCGSM::~MCGSM() {
}



void MCGSM::initialize(const MatrixXd& input, const MatrixXd& output, Parameters params) {
	MatrixXd covXX = covariance(input);
	MatrixXd covXY = covariance(input, output);

	MatrixXd whitening = SelfAdjointEigenSolver<MatrixXd>(covXX).operatorInverseSqrt();

	mScales = sampleNormal(mNumComponents, mNumScales) / 20.;
	mWeights = ArrayXXd::Random(mNumComponents, mNumFeatures).abs() / 100. + 0.01;
	mFeatures = whitening.transpose() * sampleNormal(mDimIn, mNumFeatures).matrix() / 100.;

	// optimal linear predictor and precision
	MatrixXd predictor = covXY.transpose() * covXX.inverse();
	MatrixXd choleskyFactor = covariance(output - predictor * input).inverse().llt().matrixL();
	vector<MatrixXd> choleskyFactors;

	for(int i = 0; i < mNumComponents; ++i) {
		mPredictors[i] = predictor + sampleNormal(mDimOut, mDimIn).matrix() / 10.;
		choleskyFactors.push_back(choleskyFactor);
	}

	setCholeskyFactors(choleskyFactors);
}



bool MCGSM::train(const MatrixXd& input, const MatrixXd& output, Parameters params) {
	if(input.rows() != mDimIn || output.rows() != mDimOut)
		throw Exception("Data has wrong dimensionality.");
	if(input.cols() != output.cols())
		throw Exception("The number of inputs and outputs should be the same.");

	// copy parameters for L-BFGS
	lbfgsfloatval_t* x = parameters(params);

	// optimization hyperparameters
	lbfgs_parameter_t paramsLBFGS;
	lbfgs_parameter_init(&paramsLBFGS);
	paramsLBFGS.max_iterations = params.maxIter;
	paramsLBFGS.m = params.numGrad;
	paramsLBFGS.epsilon = params.threshold;
	paramsLBFGS.linesearch = LBFGS_LINESEARCH_MORETHUENTE;
 	paramsLBFGS.max_linesearch = 100;
 	paramsLBFGS.ftol = 1e-4;
 	paramsLBFGS.xtol = 1e-32;

	// wrap additional arguments
	ParamsLBFGS instance;
	instance.first.first = this;
	instance.first.second = &params;
	instance.second.first = &input;
	instance.second.second = &output;

	// start LBFGS optimization
	if(params.maxIter > 0)
		std::cout << lbfgs(numParameters(params), x, 0, &evaluateLBFGS, &callbackLBFGS, &instance, &paramsLBFGS) << std::endl;

	// copy parameters back
	setParameters(x, params);

	// free memory used by LBFGS
	lbfgs_free(x);

	return true;
}



double MCGSM::checkGradient(
	const MatrixXd& input,
	const MatrixXd& output,
	double epsilon,
	Parameters params) const
{
	if(input.rows() != mDimIn || output.rows() != mDimOut)
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
	ParamsLBFGS instance;
	instance.first.first = this;
	instance.first.second = &params;
	instance.second.first = &input;
	instance.second.second = &output;

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



double MCGSM::checkPerformance(
	const MatrixXd& input,
	const MatrixXd& output,
	int repetitions,
	Parameters params) const
{
	if(input.rows() != mDimIn || output.rows() != mDimOut)
		throw Exception("Data has wrong dimensionality.");
	if(input.cols() != output.cols())
		throw Exception("The number of inputs and outputs should be the same.");

	// request memory for LBFGS and copy parameters
	lbfgsfloatval_t* x = parameters(params);

	// optimization hyperparameters
	lbfgs_parameter_t paramsLBFGS;
	lbfgs_parameter_init(&paramsLBFGS);
	paramsLBFGS.max_iterations = params.maxIter;
	paramsLBFGS.m = params.numGrad;

	// wrap additional arguments
	ParamsLBFGS instance;
	instance.first.first = this;
	instance.first.second = &params;
	instance.second.first = &input;
	instance.second.second = &output;

	// measure time it takes to evaluate gradient
	lbfgsfloatval_t* g = lbfgs_malloc(numParameters(params));
	timeval from, to;

	gettimeofday(&from, 0);
	for(int i = 0; i < repetitions; ++i)
		evaluateLBFGS(&instance, x, g, 0, 0.);
	gettimeofday(&to, 0);

	// free memory used by LBFGS
	lbfgs_free(x);

	return (to.tv_sec + to.tv_usec / 1E6 - from.tv_sec - from.tv_usec / 1E6) / repetitions;
}



MatrixXd MCGSM::sample(const MatrixXd& input) const {
	// initialize samples with Gaussian noise
	MatrixXd output = sampleNormal(mDimOut, input.cols());

	ArrayXXd featuresOutput = mFeatures.transpose() * input;
	MatrixXd weightsSqr = mWeights.square();
	ArrayXXd weightsOutput = weightsSqr * featuresOutput.square().matrix();
	ArrayXXd scalesExp = mScales.exp();

	#pragma omp parallel for
	for(int k = 0; k < input.cols(); ++k) {
		// compute joint distribution over components and scales
		ArrayXXd pmf = (mPriors - scalesExp.colwise() * weightsOutput.col(k) / 2.).exp();
		pmf /= pmf.sum();

		// sample component and scale
		double urand = static_cast<double>(rand()) / (static_cast<long>(RAND_MAX) + 1l);
		double cdf;
		int l = 0;

		for(cdf = pmf(0, 0); cdf < urand; cdf += pmf(l / mNumScales, l % mNumScales))
			l++;

		// component and scale index
		int i = l / mNumScales;
		int j = l % mNumScales;

		// apply precision matrix
		mCholeskyFactors[i].transpose().triangularView<Eigen::Upper>().solveInPlace(output.col(k));

		// apply scale and add mean
		output.col(k) /= sqrt(scalesExp(i, j));
		output.col(k) += mPredictors[i] * input.col(k);
	}

	return output;
}



MatrixXd MCGSM::sample(const MatrixXd& input, const Array<int, 1, Dynamic>& labels) const {
	if(input.rows() != mDimIn)
		throw Exception("Data has wrong dimensionality.");
	if(input.cols() != labels.cols())
		throw Exception("The number of inputs and labels should be the same.");

	MatrixXd output = sampleNormal(mDimOut, input.cols());

	ArrayXXd featuresOutput = mFeatures.transpose() * input;
	MatrixXd scalesExp = mScales.exp();
	MatrixXd weightsSqr = mWeights.square();

	#pragma omp parallel for
	for(int i = 0; i < input.cols(); ++i) {
		// distribution over scales
		ArrayXd pmf = mPriors.row(labels[i]).matrix() - scalesExp.row(labels[i])
			* (weightsSqr.row(labels[i]) * featuresOutput.col(i).square().matrix()) / 2.;
		pmf = (pmf - logSumExp(pmf)[0]).exp();

		// sample scale
		double urand = static_cast<double>(rand()) / (static_cast<long>(RAND_MAX) + 1l);
		double cdf;
		int j = 0;

		for(cdf = pmf(0); cdf < urand; cdf += pmf(j))
			++j;

		// apply precision matrix
		mCholeskyFactors[labels[i]].transpose().triangularView<Eigen::Upper>().solveInPlace(output.col(i));

		// apply scale
		output.col(i) /= sqrt(scalesExp(labels[i], j));

		// add predicted mean
		output.col(i) += mPredictors[labels[i]] * input.col(i);
	}

	return output;
}



MatrixXd MCGSM::reconstruct(const MatrixXd& input, const MatrixXd& output) const {
	// reconstruct output from labels
	return sample(input, samplePosterior(input, output));
}



Array<int, 1, Dynamic> MCGSM::samplePrior(const MatrixXd& input) const {
	if(input.rows() != mDimIn)
		throw Exception("Data has wrong dimensionality.");

	Array<int, 1, Dynamic> labels(input.cols());
	ArrayXXd pmf = prior(input);

	#pragma omp parallel for
	for(int j = 0; j < input.cols(); ++j) {
		int i = 0;
		double urand = static_cast<double>(rand()) / (static_cast<long>(RAND_MAX) + 1l);
		double cdf;

		// compute index
		for(cdf = pmf(0, j); cdf < urand; cdf += pmf(i, j))
			++i;

		labels[j] = i;
	}

	return labels;
}



Array<int, 1, Dynamic> MCGSM::samplePosterior(const MatrixXd& input, const MatrixXd& output) const {
	if(input.rows() != mDimIn || output.rows() != mDimOut)
		throw Exception("Data has wrong dimensionality.");
	if(input.cols() != output.cols())
		throw Exception("The number of inputs and outputs should be the same.");

	Array<int, 1, Dynamic> labels(input.cols());
	ArrayXXd pmf = posterior(input, output);

	#pragma omp parallel for
	for(int j = 0; j < input.cols(); ++j) {
		int i = 0;
		double urand = static_cast<double>(rand()) / (static_cast<long>(RAND_MAX) + 1l);
		double cdf;

		// compute index
		for(cdf = pmf(0, j); cdf < urand; cdf += pmf(i, j))
			++i;

		labels[j] = i;
	}

	return labels;
}



ArrayXXd MCGSM::prior(const MatrixXd& input) const {
	if(input.rows() != mDimIn)
		throw Exception("Data has wrong dimensionality.");

	ArrayXXd prior(mNumComponents, input.cols());

	ArrayXXd featuresOutput = mFeatures.transpose() * input;
	MatrixXd weightsOutput = mWeights.square().matrix() * featuresOutput.square().matrix();
	MatrixXd scalesExp = mScales.exp().transpose();

	#pragma omp parallel for
	for(int i = 0; i < mNumComponents; ++i) {
		// compute unnormalized posterior
		ArrayXXd negEnergy = -scalesExp.col(i) / 2. * weightsOutput.row(i);
		negEnergy.colwise() += mPriors.row(i).transpose();

		// marginalize out scales
		prior.row(i) = logSumExp(negEnergy);
	}

	// return normalized prior
	return (prior.rowwise() - logSumExp(prior)).exp();
}



ArrayXXd MCGSM::posterior(const MatrixXd& input, const MatrixXd& output) const {
	if(input.rows() != mDimIn || output.rows() != mDimOut)
		throw Exception("Data has wrong dimensionality.");
	if(input.cols() != output.cols())
		throw Exception("The number of inputs and outputs should be the same.");

	ArrayXXd posterior(mNumComponents, input.cols());

	ArrayXXd featuresOutput = mFeatures.transpose() * input;
	MatrixXd weightsOutput = mWeights.square().matrix() * featuresOutput.square().matrix();
	MatrixXd scalesExp = mScales.array().exp().transpose();

	#pragma omp parallel for
	for(int i = 0; i < mNumComponents; ++i) {
		Matrix<double, 1, Dynamic> errorSqr = 
			(mCholeskyFactors[i].transpose() * (output - mPredictors[i] * input)).colwise().squaredNorm();

		// compute unnormalized posterior
		ArrayXXd negEnergy = -scalesExp.col(i) / 2. * (weightsOutput.row(i) + errorSqr);

		// normalization constants of experts
		double logDet = mCholeskyFactors[i].diagonal().array().abs().log().sum();
		ArrayXd logPartf = mDimOut * mScales.row(i).array() / 2. + logDet;
		negEnergy.colwise() += mPriors.row(i).transpose() + logPartf;

		// marginalize out scales
		posterior.row(i) = logSumExp(negEnergy);
	}

	// return normalized prior
	return (posterior.rowwise() - logSumExp(posterior)).exp();
}



Array<double, 1, Dynamic> MCGSM::logLikelihood(const MatrixXd& input, const MatrixXd& output) const {
	if(input.rows() != mDimIn || output.rows() != mDimOut)
		throw Exception("Data has wrong dimensionality.");
	if(input.cols() != output.cols())
		throw Exception("The number of inputs and outputs should be the same.");

	ArrayXXd logLikelihood(mNumComponents, input.cols());
	ArrayXXd normConsts(mNumComponents, input.cols());

	ArrayXXd featuresOutput = mFeatures.transpose() * input;
	MatrixXd weightsOutput = mWeights.square().matrix() * featuresOutput.square().matrix();
	MatrixXd scalesExp = mScales.array().exp().transpose();

	#pragma omp parallel for
	for(int i = 0; i < mNumComponents; ++i) {
		// compute gate energy
		ArrayXXd negEnergy = -scalesExp.col(i) / 2. * weightsOutput.row(i);
		negEnergy.colwise() += mPriors.row(i).transpose();

		// normalization constants of gates
		normConsts.row(i) = logSumExp(negEnergy);

		// expert energy
		Matrix<double, 1, Dynamic> errorSqr = 
			(mCholeskyFactors[i].transpose() * (output - mPredictors[i] * input)).colwise().squaredNorm();
		negEnergy -= (scalesExp.col(i) / 2. * errorSqr).array();

		// normalization constants of experts
		double logDet = mCholeskyFactors[i].diagonal().array().abs().log().sum();
		ArrayXd logPartf = mDimOut * mScales.row(i).array() / 2. +
			logDet - mDimOut / 2. * log(2. * PI);
		negEnergy.colwise() += logPartf;

		// marginalize out scales
		logLikelihood.row(i) = logSumExp(negEnergy);
	}

	// marginalize out components
	return logSumExp(logLikelihood) - logSumExp(normConsts);
}



lbfgsfloatval_t* MCGSM::parameters(const Parameters& params) const {
	lbfgsfloatval_t* x = lbfgs_malloc(numParameters(params));

	int k = 0;
	if(params.trainPriors)
		for(int i = 0; i < mPriors.size(); ++i, ++k)
			x[k] = mPriors.data()[i];
	if(params.trainScales)
		for(int i = 0; i < mScales.size(); ++i, ++k)
			x[k] = mScales.data()[i];
	if(params.trainWeights)
		for(int i = 0; i < mWeights.size(); ++i, ++k)
			x[k] = mWeights.data()[i];
	if(params.trainFeatures)
		for(int i = 0; i < mFeatures.size(); ++i, ++k)
			x[k] = mFeatures.data()[i];
	if(params.trainCholeskyFactors)
		for(int i = 0; i < mCholeskyFactors.size(); ++i)
			for(int m = 1; m < mDimOut; ++m)
				for(int n = 0; n <= m; ++n, ++k)
					x[k] = mCholeskyFactors[i](m, n);
	if(params.trainPredictors)
		for(int i = 0; i < mPredictors.size(); ++i)
			for(int j = 0; j < mPredictors[i].size(); ++j, ++k)
				x[k] = mPredictors[i].data()[j];

	return x;
}



void MCGSM::setParameters(const lbfgsfloatval_t* x, const Parameters& params) {
	int offset = 0;

	if(params.trainPriors) {
		mPriors = MatrixLBFGS(const_cast<double*>(x), mNumComponents, mNumScales);
		offset += mPriors.size();
	}

	if(params.trainScales) {
		mScales = MatrixLBFGS(const_cast<double*>(x) + offset, mNumComponents, mNumScales);
		offset += mScales.size();
	}

	if(params.trainWeights) {
		mWeights = MatrixLBFGS(const_cast<double*>(x) + offset, mNumComponents, mNumFeatures);
		offset += mWeights.size();
	}

	if(params.trainFeatures) {
		mFeatures = MatrixLBFGS(const_cast<double*>(x) + offset, mDimIn, mNumFeatures);
		offset += mFeatures.size();
	}

	if(params.trainCholeskyFactors)
		for(int i = 0; i < mNumComponents; ++i) {
			mCholeskyFactors[i].setZero();
			mCholeskyFactors[i](0, 0) = 1.;
			for(int m = 1; m < mDimOut; ++m)
				for(int n = 0; n <= m; ++n, ++offset)
					mCholeskyFactors[i](m, n) = x[offset];
		}

	if(params.trainPredictors)
		for(int i = 0; i < mNumComponents; ++i) {
			mPredictors[i] = MatrixLBFGS(const_cast<double*>(x) + offset, mDimOut, mDimIn);
			offset += mPredictors[i].size();
		}
}



double MCGSM::computeGradient(
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

	MatrixLBFGS priors(params.trainPriors ? y : const_cast<double*>(mPriors.data()), mNumComponents, mNumScales);
	MatrixLBFGS priorsGrad = MatrixLBFGS(g, mNumComponents, mNumScales);
	if(params.trainPriors)
		offset += priors.size();

	MatrixLBFGS scales(params.trainScales ? y + offset : const_cast<double*>(mScales.data()), mNumComponents, mNumScales);
	MatrixLBFGS scalesGrad(g + offset, mNumComponents, mNumScales);
	if(params.trainScales)
		offset += scales.size();

	MatrixLBFGS weights(params.trainWeights ? y + offset : const_cast<double*>(mWeights.data()), mNumComponents, mNumFeatures);
	MatrixLBFGS weightsGrad(g + offset, mNumComponents, mNumFeatures);
	if(params.trainWeights)
		offset += weights.size();

	MatrixLBFGS features(params.trainFeatures ? y + offset : const_cast<double*>(mFeatures.data()), mDimIn, mNumFeatures);
	MatrixLBFGS featuresGrad(g + offset, mDimIn, mNumFeatures);
	if(params.trainFeatures)
		offset += features.size();

	vector<MatrixXd> choleskyFactors;
	vector<MatrixXd> choleskyFactorsGrad;

	// store memory position of Cholesky factors for later
	int cholFacOffset = offset;

	if(params.trainCholeskyFactors)
		for(int i = 0; i < mNumComponents; ++i) {
			choleskyFactors.push_back(MatrixXd::Zero(mDimOut, mDimOut));
			choleskyFactorsGrad.push_back(MatrixXd::Zero(mDimOut, mDimOut));
			choleskyFactors[i](0, 0) = 1.;
			for(int m = 1; m < mDimOut; ++m)
				for(int n = 0; n <= m; ++n, ++offset)
					choleskyFactors[i](m, n) = x[offset];
		}
	else
		for(int i = 0; i < mNumComponents; ++i)
			choleskyFactors.push_back(mCholeskyFactors[i]);

	vector<MatrixLBFGS> predictors;
	vector<MatrixLBFGS> predictorsGrad;

	if(params.trainPredictors)
		for(int i = 0; i < mNumComponents; ++i) {
			predictors.push_back(MatrixLBFGS(y + offset, mDimOut, mDimIn));
			predictorsGrad.push_back(MatrixLBFGS(g + offset, mDimOut, mDimIn));
			offset += predictors[i].size();
		}
	else
		for(int i = 0; i < mNumComponents; ++i)
			predictors.push_back(MatrixLBFGS(const_cast<double*>(mPredictors[i].data()), mDimOut, mDimIn));

	if(g) {
		// initialize gradients
		if(params.trainFeatures)
			featuresGrad.setZero();
		if(params.trainWeights)
			weightsGrad.setZero();
		if(params.trainPriors)
			priorsGrad.setZero();
		if(params.trainScales)
			scalesGrad.setZero();
		if(params.trainPredictors)
			for(int i = 0; i < mNumComponents; ++i)
				predictorsGrad[i].setZero();
	}

	// split data into batches for better performance
	int numData = static_cast<int>(inputCompl.cols());
	int batchSize = min(params.batchSize, numData);

	for(int b = 0; b < inputCompl.cols(); b += batchSize) {
		const MatrixXd input = inputCompl.middleCols(b, min(batchSize, numData - b));
		const MatrixXd output = outputCompl.middleCols(b, min(batchSize, numData - b));

		// compute unnormalized posterior
		MatrixXd featureOutput = features.transpose() * input;
		MatrixXd featureOutputSqr = featureOutput.array().square();
		MatrixXd weightsSqr = weights.array().square();
		MatrixXd weightsOutput = weightsSqr * featureOutputSqr;

		// containers for intermediate results
		vector<ArrayXXd> logPosteriorIn(mNumComponents);
		vector<ArrayXXd> logPosteriorOut(mNumComponents);
		vector<MatrixXd> predError(mNumComponents);
		vector<Array<double, 1, Dynamic> > predErrorSqNorm(mNumComponents);
		vector<MatrixXd> scalesExp(mNumComponents);

		// partial normalization constants
		ArrayXXd logNormInScales(mNumComponents, input.cols());
		ArrayXXd logNormOutScales(mNumComponents, input.cols());

		#pragma omp parallel for
		for(int i = 0; i < mNumComponents; ++i) {
			scalesExp[i] = scales.row(i).transpose().array().exp();

			MatrixXd negEnergyGate = -scalesExp[i] / 2. * weightsOutput.row(i);
			negEnergyGate.colwise() += priors.row(i).transpose();

			predError[i] = output - predictors[i] * input;
			predErrorSqNorm[i] = (choleskyFactors[i].transpose() * predError[i]).colwise().squaredNorm();

			MatrixXd negEnergyExpert = -scalesExp[i] / 2. * predErrorSqNorm[i].matrix();

			// normalize expert energy
			double logDet = choleskyFactors[i].diagonal().array().abs().log().sum();
			VectorXd logPartf = mDimOut / 2. * scales.row(i).transpose().array()
				+ logDet - mDimOut / 2. * log(2. * PI);

			negEnergyExpert.colwise() += logPartf;

			// unnormalized posterior
			logPosteriorIn[i] = negEnergyGate;
			logPosteriorOut[i] = negEnergyGate + negEnergyExpert;

			// compute normalization constants for posterior over scales
			logNormInScales.row(i) = logSumExp(logPosteriorIn[i]);
			logNormOutScales.row(i) = logSumExp(logPosteriorOut[i]);
		}

		Array<double, 1, Dynamic> logNormIn;
		Array<double, 1, Dynamic> logNormOut;

		// compute normalization constants
		#pragma omp parallel sections
		{
			#pragma omp section
			logNormIn = logSumExp(logNormInScales);
			#pragma omp section
			logNormOut = logSumExp(logNormOutScales);
		}

		// predictive probability
		logLik += (logNormOut - logNormIn).sum();

		if(!g)
			// don't compute gradients
			continue;

		// compute gradients
		#pragma omp parallel for
		for(int i = 0; i < mNumComponents; ++i) {
			// normalize posterior
			logPosteriorIn[i].rowwise() -= logNormIn;
			logPosteriorOut[i].rowwise() -= logNormOut;

			ArrayXXd posteriorIn = logPosteriorIn[i].exp();
			ArrayXXd posteriorOut = logPosteriorOut[i].exp();
			MatrixXd posteriorDiff = posteriorIn - posteriorOut;

			// gradient of prior variables
			if(params.trainPriors)
				priorsGrad.row(i) += posteriorDiff.rowwise().sum();// + 1. * priors;

			Array<double, 1, Dynamic> tmp0 = -scalesExp[i].transpose() * posteriorDiff;

			if(params.trainWeights) {
				Array<double, 1, Dynamic> tmp1 = (featureOutputSqr.array().rowwise() * tmp0).rowwise().sum();

				// gradient of weights
				weightsGrad.row(i) += (tmp1 * weights.row(i).array()).matrix();
			}

			Array<double, 1, Dynamic> tmp3 = posteriorOut.rowwise().sum();

			// gradient of scale variables
			if(params.trainScales) {
				Array<double, 1, Dynamic> tmp2 = (posteriorDiff.array().rowwise() * weightsOutput.row(i).array()).rowwise().sum();
				Array<double, 1, Dynamic> tmp4 = (posteriorOut.rowwise() * predErrorSqNorm[i]).rowwise().sum();

				scalesGrad.row(i) += (
					tmp4 * scales.row(i).array().exp() / 2. - 
					tmp3 * mDimOut / 2. -
					tmp2 * scales.row(i).array().exp() / 2.).matrix();
			}

			// partial gradient of features
			if(params.trainFeatures) {
				MatrixXd tmp5 = input.array().rowwise() * tmp0;
				ArrayXXd tmp6 = tmp5 * featureOutput.transpose();

				#pragma omp critical
				featuresGrad += (tmp6.rowwise() * weightsSqr.row(i).array()).matrix();
			}

			Array<double, 1, Dynamic> tmp7 = scalesExp[i].transpose() * posteriorOut.matrix();
			MatrixXd tmp8 = predError[i].array().rowwise() * tmp7;

			// gradient of cholesky factor
			if(params.trainCholeskyFactors) {
				MatrixXd tmp9 = choleskyFactors[i].diagonal().cwiseInverse().asDiagonal();

				choleskyFactorsGrad[i] += tmp8 * predError[i].transpose() * choleskyFactors[i]
					- tmp3.sum() * tmp9;
			}

			// gradient of linear predictor
			if(params.trainPredictors) {
				MatrixXd precision = choleskyFactors[i] * choleskyFactors[i].transpose();

				predictorsGrad[i] -= precision * tmp8 * input.transpose();
			}
		}
	}

	double normConst = inputCompl.cols() * log(2.);

	if(g) {
		// write back gradients of Cholesky factors
		if(params.trainCholeskyFactors)
			for(int i = 0; i < mNumComponents; ++i)
				for(int m = 1; m < mDimOut; ++m)
					for(int n = 0; n <= m; ++n, ++cholFacOffset)
						g[cholFacOffset] = choleskyFactorsGrad[i](m, n);

		for(int i = 0; i < offset; ++i)
			g[i] /= normConst;
	}

	// return average log-likelihood
	return -logLik / normConst; 
}
