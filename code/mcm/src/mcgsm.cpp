#include "mcgsm.h"
#include "utils.h"
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
//	const MCGSM& mcgsm = *static_cast<ParamsLBFGS*>(instance)->first.first;
	const MCGSM::Parameters& params = *static_cast<ParamsLBFGS*>(instance)->first.second;

	if(params.verbosity > 0)
		cout << setw(6) << iteration << setw(10) << setprecision(5) << fx << endl;

	return 0;
}



static lbfgsfloatval_t evaluateLBFGS(
	void* instance, 
	const lbfgsfloatval_t* x, 
	lbfgsfloatval_t* g, 
	int, double) 
{
	// unpack user data
	const MCGSM& mcgsm = *static_cast<ParamsLBFGS*>(instance)->first.first;
	const MCGSM::Parameters& params = *static_cast<ParamsLBFGS*>(instance)->first.second;
	const MatrixXd& inputCompl = *static_cast<ParamsLBFGS*>(instance)->second.first;
	const MatrixXd& outputCompl = *static_cast<ParamsLBFGS*>(instance)->second.second;

	// average log-likelihood
	double logLik = 0.;

	// interpret memory for parameters and gradients
	lbfgsfloatval_t* y = const_cast<lbfgsfloatval_t*>(x);

	int offset = 0;

	MatrixLBFGS priors = MatrixLBFGS(y, mcgsm.numComponents(), mcgsm.numScales());
	MatrixLBFGS priorsGrad(g, mcgsm.numComponents(), mcgsm.numScales());
	offset += priors.size();

	MatrixLBFGS scales(y + offset, mcgsm.numComponents(), mcgsm.numScales());
	MatrixLBFGS scalesGrad(g + offset, mcgsm.numComponents(), mcgsm.numScales());
	offset += scales.size();

	MatrixLBFGS weights(y + offset, mcgsm.numComponents(), mcgsm.numFeatures());
	MatrixLBFGS weightsGrad(g + offset, mcgsm.numComponents(), mcgsm.numFeatures());
	offset += weights.size();

	MatrixLBFGS features(y + offset, mcgsm.dimIn(), mcgsm.numFeatures());
	MatrixLBFGS featuresGrad(g + offset, mcgsm.dimIn(), mcgsm.numFeatures());
	offset += features.size();

	vector<MatrixXd> choleskyFactors;
	vector<MatrixXd> choleskyFactorsGrad;

	// store memory position of Cholesky factors for later
	int cholFacOffset = offset;

	for(int i = 0; i < mcgsm.numComponents(); ++i) {
		choleskyFactors.push_back(MatrixXd::Zero(mcgsm.dimOut(), mcgsm.dimOut()));
		choleskyFactorsGrad.push_back(MatrixXd::Zero(mcgsm.dimOut(), mcgsm.dimOut()));
		choleskyFactors[i](0, 0) = 1.;
		for(int m = 1; m < mcgsm.dimOut(); ++m)
			for(int n = 0; n <= m; ++n, ++offset)
				choleskyFactors[i](m, n) = x[offset];
	}

	vector<MatrixLBFGS> predictors;
	vector<MatrixLBFGS> predictorsGrad;

	for(int i = 0; i < mcgsm.numComponents(); ++i) {
		predictors.push_back(MatrixLBFGS(y + offset, mcgsm.dimOut(), mcgsm.dimIn()));
		predictorsGrad.push_back(MatrixLBFGS(g + offset, mcgsm.dimOut(), mcgsm.dimIn()));
		offset += predictors[i].size();
	}

	if(g) {
		// initialize gradients
		featuresGrad.setZero();
		weightsGrad.setZero();
		priorsGrad.setZero();
		scalesGrad.setZero();

		for(int i = 0; i < mcgsm.numComponents(); ++i)
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
		vector<ArrayXXd> logPosteriorIn(mcgsm.numComponents());
		vector<ArrayXXd> logPosteriorOut(mcgsm.numComponents());
		vector<MatrixXd> predError(mcgsm.numComponents());
		vector<Array<double, 1, Dynamic> > predErrorSqNorm(mcgsm.numComponents());
		vector<MatrixXd> scalesSqr(mcgsm.numComponents());

		// partial normalization constants
		ArrayXXd logNormInScales(mcgsm.numComponents(), input.cols());
		ArrayXXd logNormOutScales(mcgsm.numComponents(), input.cols());

		#pragma omp parallel for
		for(int i = 0; i < mcgsm.numComponents(); ++i) {
			scalesSqr[i] = scales.row(i).transpose().array().square();

			MatrixXd negEnergyGate = -scalesSqr[i] / 2. * weightsOutput.row(i);
			negEnergyGate.colwise() += priors.row(i).transpose();

			predError[i] = output - predictors[i] * input;
			predErrorSqNorm[i] = (choleskyFactors[i].transpose() * predError[i]).colwise().squaredNorm();

			MatrixXd negEnergyExpert = -scalesSqr[i] / 2. * predErrorSqNorm[i].matrix();

			// normalize expert energy
			double logDet = choleskyFactors[i].diagonal().array().abs().log().sum();
			VectorXd logPartf = mcgsm.dimOut() * scales.row(i).transpose().array().abs().log()
				+ logDet - mcgsm.dimOut() / 2. * log(2. * PI);

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
		for(int i = 0; i < mcgsm.numComponents(); ++i) {
			// normalize posterior
			logPosteriorIn[i].rowwise() -= logNormIn;
			logPosteriorOut[i].rowwise() -= logNormOut;

			ArrayXXd posteriorIn = logPosteriorIn[i].exp();
			ArrayXXd posteriorOut = logPosteriorOut[i].exp();

			MatrixXd posteriorDiff = posteriorIn - posteriorOut;

			// gradient of prior variables
			priorsGrad.row(i) += posteriorDiff.rowwise().sum();

			Array<double, 1, Dynamic> tmp0 = -scalesSqr[i].transpose() * posteriorDiff;
			Array<double, 1, Dynamic> tmp1 = (featureOutputSqr.array().rowwise() * tmp0).rowwise().sum();

			// gradient of weights
 			weightsGrad.row(i) += (tmp1 * weights.row(i).array()).matrix();

			Array<double, 1, Dynamic> tmp2 = (posteriorDiff.array().rowwise() * weightsOutput.row(i).array()).rowwise().sum();
			Array<double, 1, Dynamic> tmp3 = posteriorOut.rowwise().sum();
			Array<double, 1, Dynamic> tmp4 = (posteriorOut.rowwise() * predErrorSqNorm[i]).rowwise().sum();

			// gradient of scale variables
 			scalesGrad.row(i) += (
 				tmp4 * scales.row(i).array() - 
 				tmp3 / scales.row(i).array() * mcgsm.dimOut() -
 				tmp2 * scales.row(i).array()).matrix();

			MatrixXd tmp5 = input.array().rowwise() * tmp0;
			ArrayXXd tmp6 = tmp5 * featureOutput.transpose();

			// partial gradient of features
			#pragma omp critical
			featuresGrad += (tmp6.rowwise() * weightsSqr.row(i).array()).matrix();

			Array<double, 1, Dynamic> tmp7 = scalesSqr[i].transpose() * posteriorOut.matrix();
			MatrixXd tmp8 = predError[i].array().rowwise() * tmp7;
			MatrixXd tmp9 = choleskyFactors[i].diagonal().cwiseInverse().asDiagonal();

			// gradient of cholesky factor
			choleskyFactorsGrad[i] += tmp8 * predError[i].transpose() * choleskyFactors[i].transpose()
				- tmp3.sum() * tmp9;

			MatrixXd precision = choleskyFactors[i] * choleskyFactors[i].transpose();

			// gradient of linear predictor
			predictorsGrad[i] -= precision * tmp8 * input.transpose();
		}
	}

	if(g) {
		// write back gradients of Cholesky factors
		for(int i = 0; i < mcgsm.numComponents(); ++i)
			for(int m = 1; m < mcgsm.dimOut(); ++m)
				for(int n = 0; n <= m; ++n, ++cholFacOffset)
					g[cholFacOffset] = choleskyFactorsGrad[i](m, n);

		for(int i = 0; i < offset; ++i)
			g[i] /= inputCompl.cols();
	}

	// return average log-likelihood
	return -logLik / inputCompl.cols();
}



MCGSM::Parameters::Parameters() {
	verbosity = 0;
	maxIter = 1000;
	threshold = 1e-5;
	numGrad = 20;
	batchSize = 2000;
};



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
	mPriors = ArrayXXd::Random(mNumComponents, mNumScales) / 10.;
	mScales = ArrayXXd::Random(mNumComponents, mNumScales).abs() / 2. + 0.75;
	mWeights = ArrayXXd::Random(mNumComponents, mNumFeatures).abs() / 10. + 0.01;
	mFeatures = ArrayXXd::Random(mDimIn, mNumFeatures) / 10.;

	for(int i = 0; i < mNumComponents; ++i) {
		mCholeskyFactors.push_back(VectorXd::Ones(mDimOut).asDiagonal());
		mPredictors.push_back(ArrayXXd::Random(mDimOut, mDimIn) / 10.);
	}
}



MCGSM::~MCGSM() {
}



void MCGSM::normalize() {
}



bool MCGSM::train(const MatrixXd& input, const MatrixXd& output, Parameters params) {
	if(input.rows() != mDimIn || output.rows() != mDimOut)
		throw Exception("Data has wrong dimensionality.");

	// copy parameters for L-BFGS
	lbfgsfloatval_t* x = parameters();

	// optimization hyperparameters
	lbfgs_parameter_t paramsLBFGS;
	lbfgs_parameter_init(&paramsLBFGS);
	paramsLBFGS.max_iterations = params.maxIter;
	paramsLBFGS.m = params.numGrad;
	paramsLBFGS.epsilon = params.threshold;

	// wrap additional arguments
	ParamsLBFGS instance;
	instance.first.first = this;
	instance.first.second = &params;
	instance.second.first = &input;
	instance.second.second = &output;

	// start LBFGS optimization
	if(params.maxIter > 0)
		lbfgs(numParameters(), x, 0, &evaluateLBFGS, &callbackLBFGS, &instance, &paramsLBFGS);

	// copy parameters back
	setParameters(x);

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

	// request memory for LBFGS and copy parameters
	lbfgsfloatval_t* x = parameters();

	int numParams = numParameters();

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

	// compute numerical gradient
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

	// request memory for LBFGS and copy parameters
	lbfgsfloatval_t* x = parameters();

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
	lbfgsfloatval_t* g = lbfgs_malloc(numParameters());
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
	return MatrixXd::Random(mDimOut, input.cols());
}



Array<double, 1, Dynamic> MCGSM::samplePosterior(const MatrixXd& input, const MatrixXd& output) const {
	return Array<double, 1, Dynamic>::Random(1, input.cols());
}



ArrayXXd MCGSM::posterior(const MatrixXd& input, const MatrixXd& output) const {
	return ArrayXXd::Random(mNumComponents, input.cols());
}



Array<double, 1, Dynamic> MCGSM::logLikelihood(const MatrixXd& input, const MatrixXd& output) const {
	return ArrayXXd::Random(1, input.cols());
}



lbfgsfloatval_t* MCGSM::parameters() const {
	lbfgsfloatval_t* x = lbfgs_malloc(numParameters());

	int k = 0;
	for(int i = 0; i < mPriors.size(); ++i, ++k)
		x[k] = mPriors.data()[i];
	for(int i = 0; i < mScales.size(); ++i, ++k)
		x[k] = mScales.data()[i];
	for(int i = 0; i < mWeights.size(); ++i, ++k)
		x[k] = mWeights.data()[i];
	for(int i = 0; i < mFeatures.size(); ++i, ++k)
		x[k] = mFeatures.data()[i];
	for(int i = 0; i < mCholeskyFactors.size(); ++i)
		for(int m = 1; m < mDimOut; ++m)
			for(int n = 0; n <= m; ++n, ++k)
				x[k] = mCholeskyFactors[i](m, n);
	for(int i = 0; i < mPredictors.size(); ++i)
		for(int j = 0; j < mPredictors[i].size(); ++j, ++k)
			x[k] = mPredictors[i].data()[j];

	return x;
}


void MCGSM::setParameters(const lbfgsfloatval_t* x) {
	int offset = 0;

	mPriors = MatrixLBFGS(const_cast<double*>(x), mNumComponents, mNumScales);
	offset += mPriors.size();

	mScales = MatrixLBFGS(const_cast<double*>(x) + offset, mNumComponents, mNumScales);
	offset += mScales.size();

	mWeights = MatrixLBFGS(const_cast<double*>(x) + offset, mNumComponents, mNumFeatures);
	offset += mWeights.size();

	mFeatures = MatrixLBFGS(const_cast<double*>(x) + offset, mDimIn, mNumFeatures);
	offset += mFeatures.size();

	for(int i = 0; i < mNumComponents; ++i) {
		mCholeskyFactors[i].setZero();
		mCholeskyFactors[i](0, 0) = 1.;
		for(int m = 1; m < mDimOut; ++m)
			for(int n = 0; n <= m; ++n, ++offset)
				mCholeskyFactors[i](m, n) = x[offset];
	}

	for(int i = 0; i < mNumComponents; ++i) {
		mPredictors[i] = MatrixLBFGS(const_cast<double*>(x) + offset, mDimOut, mDimIn);
		offset += mPredictors[i].size();
	}
}
