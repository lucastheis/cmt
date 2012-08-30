#include "mcgsm.h"
#include "utils.h"
#include "lbfgs.h"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <utility>

using namespace std;

typedef pair<MCGSM*, pair<const MatrixXd*, const MatrixXd*> > ParamsLBFGS;
typedef Map<Matrix<lbfgsfloatval_t, Dynamic, Dynamic> > MatrixLBFGS;

static lbfgsfloatval_t evaluateLBFGS(void* instance, const lbfgsfloatval_t* x, lbfgsfloatval_t* g, int, double) {
	// unpack user data
	MCGSM* mcgsm = static_cast<ParamsLBFGS*>(instance)->first;
	const MatrixXd& input = *static_cast<ParamsLBFGS*>(instance)->second.first;
	const MatrixXd& output = *static_cast<ParamsLBFGS*>(instance)->second.second;

	// interpret memory for parameters and gradients
	lbfgsfloatval_t* y = const_cast<lbfgsfloatval_t*>(x);

	int offset = 0;

	MatrixLBFGS priors = MatrixLBFGS(y, mcgsm->numComponents(), mcgsm->numScales());
	MatrixLBFGS priorsGrad(g, mcgsm->numComponents(), mcgsm->numScales());
	offset += priors.size();

	MatrixLBFGS scales(y + offset, mcgsm->numComponents(), mcgsm->numScales());
	MatrixLBFGS scalesGrad(g + offset, mcgsm->numComponents(), mcgsm->numScales());
	offset += scales.size();

	MatrixLBFGS weights(y + offset, mcgsm->numComponents(), mcgsm->numFeatures());
	MatrixLBFGS weightsGrad(g + offset, mcgsm->numComponents(), mcgsm->numFeatures());
	offset += weights.size();

	MatrixLBFGS features(y + offset, mcgsm->dimIn(), mcgsm->numFeatures());
	MatrixLBFGS featuresGrad(g + offset, mcgsm->dimIn(), mcgsm->numFeatures());
	offset += features.size();

	vector<MatrixXd> choleskyFactors;
	vector<MatrixXd> choleskyFactorsGrad;

	// store memory position of Cholesky factors for later
	int cholFacOffset = offset;

	for(int i = 0; i < mcgsm->numComponents(); ++i) {
		choleskyFactors.push_back(MatrixXd::Zero(mcgsm->dimOut(), mcgsm->dimOut()));
		choleskyFactorsGrad.push_back(MatrixXd::Zero(mcgsm->dimOut(), mcgsm->dimOut()));
		choleskyFactors[i](0, 0) = 1.;
		for(int m = 1; m < mcgsm->dimOut(); ++m)
			for(int n = 0; n <= m; ++n, ++offset)
				choleskyFactors[i](m, n) = x[offset];
	}

	vector<MatrixLBFGS> predictors;
	vector<MatrixLBFGS> predictorsGrad;

	for(int i = 0; i < mcgsm->numComponents(); ++i) {
		predictors.push_back(MatrixLBFGS(y + offset, mcgsm->dimOut(), mcgsm->dimIn()));
		predictorsGrad.push_back(MatrixLBFGS(g + offset, mcgsm->dimOut(), mcgsm->dimIn()));
		offset += predictors[i].size();
	}

	// compute unnormalized posterior
	MatrixXd featureOutput = features.transpose() * input;
	MatrixXd featureOutputSqr = featureOutput.array().square();
	MatrixXd weightsSqr = weights.array().square();
	MatrixXd weightsOutput = weightsSqr * featureOutputSqr;

	vector<ArrayXXd> logPosteriorIn;
	vector<ArrayXXd> logPosteriorOut;
	vector<MatrixXd> predError;
	vector<Array<double, 1, Dynamic> > predErrorSqNorm;
	vector<MatrixXd> featuresGradScales;

	for(int i = 0; i < mcgsm->numComponents(); ++i) {
		logPosteriorIn.push_back(ArrayXXd(mcgsm->numScales(), input.cols()));
		logPosteriorOut.push_back(ArrayXXd(mcgsm->numScales(), input.cols()));
		predError.push_back(MatrixXd(mcgsm->dimOut(), input.cols()));
		predErrorSqNorm.push_back(Array<double, 1, Dynamic>(input.cols()));
		featuresGradScales.push_back(MatrixXd(mcgsm->dimIn(), mcgsm->numFeatures()));
	}

	// partial normalization constants
	ArrayXXd logNormInScales(mcgsm->numComponents(), input.cols());
	ArrayXXd logNormOutScales(mcgsm->numComponents(), input.cols());

	#pragma omp parallel for
	for(int i = 0; i < mcgsm->numComponents(); ++i) {
		MatrixXd scalesSqr = scales.row(i).transpose().array().square();
		MatrixXd negEnergyGate = -scalesSqr / 2. * weightsOutput.row(i);
		negEnergyGate.colwise() += priors.row(i).transpose();

		predError[i] = output - predictors[i] * input;
		predErrorSqNorm[i] = (choleskyFactors[i].transpose() * predError[i]).colwise().squaredNorm();

		MatrixXd negEnergyExpert = -scalesSqr / 2. * predErrorSqNorm[i].matrix();

		// normalize expert energy
		double logDet = choleskyFactors[i].diagonal().array().abs().log().sum();
		VectorXd logPartf = mcgsm->dimOut() * scales.row(i).transpose().array().abs().log()
			+ logDet - mcgsm->dimOut() / 2. * log(2. * PI);

		negEnergyExpert.colwise() += logPartf;

		// unnormalized posterior
		logPosteriorIn[i] = negEnergyGate;
		logPosteriorOut[i] = negEnergyGate + negEnergyExpert;

		// compute normalization constants for posterior over scales
		logNormInScales.row(i) = logSumExp(logPosteriorIn[i]);
		logNormOutScales.row(i) = logSumExp(logPosteriorOut[i]);
	}

	// compute normalization constants
	Array<double, 1, Dynamic> logNormIn = logSumExp(logNormInScales);
	Array<double, 1, Dynamic> logNormOut = logSumExp(logNormOutScales);

	// predictive probability
	Array<double, 1, Dynamic> logProb = logNormOut - logNormIn;

	if(!g)
		// don't compute gradients; return average log-likelihood
		return -logProb.mean();

	// compute gradients
	#pragma omp parallel for
	for(int i = 0; i < mcgsm->numComponents(); ++i) {
		MatrixXd scalesSqr = scales.row(i).transpose().array().square();

		// normalize posterior
		logPosteriorIn[i].rowwise() -= logNormIn;
		logPosteriorOut[i].rowwise() -= logNormOut;

		MatrixXd posteriorDiff = logPosteriorIn[i].exp() - logPosteriorOut[i].exp();

		// gradient of prior variables
		priorsGrad.row(i) = posteriorDiff.rowwise().mean();

		Array<double, 1, Dynamic> tmp0 = -scalesSqr.transpose() * posteriorDiff;
		Array<double, 1, Dynamic> tmp2 = (featureOutputSqr.array().rowwise() * tmp0).rowwise().mean();

		// gradient of weights
		weightsGrad.row(i) = tmp2 * weights.row(i).array();

		Array<double, 1, Dynamic> tmp3 = -(posteriorDiff.array().rowwise() * weightsOutput.row(i).array()).rowwise().mean();
		Array<double, 1, Dynamic> tmp4 = -logPosteriorOut[i].exp().rowwise().mean();
		Array<double, 1, Dynamic> tmp5 = (logPosteriorOut[i].exp().rowwise() * predErrorSqNorm[i]).rowwise().mean();

		// gradient of scale variables
		scalesGrad.row(i) =
			tmp3 * scales.row(i).array() +
			tmp4 / scales.row(i).array() * mcgsm->dimOut() +
			tmp5 * scales.row(i).array();

		MatrixXd tmp6 = input.array().rowwise() * tmp0;
		ArrayXXd tmp7 = tmp6 * featureOutput.transpose() / input.cols();

		// gradient of linear features
		featuresGradScales[i] = tmp7.rowwise() * weightsSqr.row(i).array();

		Array<double, 1, Dynamic> tmp8 = scalesSqr.transpose() * logPosteriorOut[i].exp().matrix();
		MatrixXd tmp9 = predError[i].array().rowwise() * tmp8;
		MatrixXd tmp10 = choleskyFactors[i].diagonal().cwiseInverse().asDiagonal();

		choleskyFactorsGrad[i] = tmp9 * predError[i].transpose() * choleskyFactors[i].transpose() / input.cols()
			+ tmp4.sum() * tmp10;

		MatrixXd precision = choleskyFactors[i] * choleskyFactors[i].transpose();

		// gradient of linear predictor
		predictorsGrad[i] = -precision * tmp9 * input.transpose() / input.cols();
	}

	featuresGrad.setZero();

	for(int i = 0; i < mcgsm->numComponents(); ++i)
		featuresGrad += featuresGradScales[i];

	// write back gradients of Cholesky factors
	for(int i = 0; i < mcgsm->numComponents(); ++i)
		for(int m = 1; m < mcgsm->dimOut(); ++m)
			for(int n = 0; n <= m; ++n, ++cholFacOffset)
				g[cholFacOffset] = choleskyFactorsGrad[i](m, n);

	// return average log-likelihood
	return -logProb.mean();
}



static bool checkLBFGS(int numParams, void* instance, const lbfgsfloatval_t* x) {
	lbfgsfloatval_t y[numParams];
	lbfgsfloatval_t g[numParams];
	lbfgsfloatval_t n[numParams];
	lbfgsfloatval_t val1;
	lbfgsfloatval_t val2;

	double epsilon = 0.00001;

	// copy parameters
	for(int i = 0; i < numParams; ++i)
		y[i] = x[i];

	// compute numerical gradient
	for(int i = 0; i < numParams; ++i) {
		y[i] = x[i] + epsilon;
		val1 = evaluateLBFGS(instance, y, 0, 0, 0.);
		y[i] = x[i] - epsilon;
		val2 = evaluateLBFGS(instance, y, 0, 0, 0.);
		y[i] = x[i];
		n[i] = (val1 - val2) / (2. * epsilon);
	}

//	double gsqr = 0.;
//	for(int i = 0; i < numParams; ++i) {
//		y[i] = x[i] - epsilon * n[i];
//		gsqr += n[i] * n[i];
//	}
//
//	val1 = evaluateLBFGS(instance, x, 0, 0, 0.);
//	val2 = evaluateLBFGS(instance, y, 0, 0, 0.);
//
//	cout << "Increase:" << val2 - val1 << endl;
//	cout << "Expected increase:" << -epsilon * gsqr << endl;

	// compute analytical gradient
	evaluateLBFGS(instance, x, g, 0, 0.);

	// unpack user data
	MCGSM* mcgsm = static_cast<ParamsLBFGS*>(instance)->first;

	int numParamsPriors = mcgsm->numComponents() * mcgsm->numScales();
	int numParamsScales = mcgsm->numComponents() * mcgsm->numScales();
	int numParamsWeights = mcgsm->numComponents() * mcgsm->numFeatures();
	int numParamsFeatures = mcgsm->dimIn() * mcgsm->numFeatures();
	int numParamsCholeskyFactors = mcgsm->numComponents() * mcgsm->dimOut() * 
		(mcgsm->dimOut() + 1) / 2 - mcgsm->numComponents();
	int numParamsPredictors = mcgsm->numComponents() * mcgsm->dimIn() * mcgsm->dimOut();
	int j = 0;
	double err;

	err = 0.;
	for(int i = 0; i < numParamsPriors; ++i, ++j)
		err += (g[j] - n[j]) * (g[j] - n[j]);
	cout << "Error in priorsGrad (" << numParamsPriors << " parameters): " << sqrt(err) << endl;

	err = 0.;
	for(int i = 0; i < numParamsScales; ++i, ++j)
		err += (g[j] - n[j]) * (g[j] - n[j]);
	cout << "Error in scalesGrad (" << numParamsScales << " parameters): " << sqrt(err) << endl;

	err = 0.;
	for(int i = 0; i < numParamsWeights; ++i, ++j)
		err += (g[j] - n[j]) * (g[j] - n[j]);
	cout << "Error in weightsGrad (" << numParamsWeights << " parameters): " << sqrt(err) << endl;

	err = 0.;
	for(int i = 0; i < numParamsFeatures; ++i, ++j)
		err += (g[j] - n[j]) * (g[j] - n[j]);
	cout << "Error in featuresGrad (" << numParamsFeatures << " parameters): " << sqrt(err) << endl;

	err = 0.;
	for(int i = 0; i < numParamsCholeskyFactors; ++i, ++j)
		err += (g[j] - n[j]) * (g[j] - n[j]);
	cout << "Error in choleskyFactorsGrad (" << numParamsCholeskyFactors << " parameters): " << sqrt(err) << endl;

	err = 0.;
	for(int i = 0; i < numParamsPredictors; ++i, ++j)
		err += (g[j] - n[j]) * (g[j] - n[j]);
	cout << "Error in predictorsGrad (" << numParamsPredictors << " parameters): " << sqrt(err) << endl;
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
	mPriors = ArrayXXd::Random(mNumComponents, mNumScales) / 10.;
//	mPriors = ArrayXXd::Zero(mNumComponents, mNumScales);
	mScales = ArrayXXd::Random(mNumComponents, mNumScales).abs() / 2. + 0.75;
//	mScales = ArrayXXd::Ones(mNumComponents, mNumScales);
	mWeights = ArrayXXd::Random(mNumComponents, mNumFeatures).abs() / 10. + 0.01;
//	mWeights = ArrayXXd::Zero(mNumComponents, mNumFeatures);
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



bool MCGSM::train(const MatrixXd& input, const MatrixXd& output, int maxIter, double tol) {
	if(input.rows() != mDimIn || output.rows() != mDimOut)
		throw Exception("Data has wrong dimensionality.");

	int numParams = mPriors.size() + mScales.size() + mWeights.size() + mFeatures.size()
		+ mNumComponents * mDimOut * (mDimOut + 1) / 2 - mNumComponents
		+ mNumComponents * mPredictors[0].size();

	// request memory for LBFGS
	lbfgsfloatval_t* x = lbfgs_malloc(numParams);

	// copy parameters
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

	// optimization hyperparameters
	lbfgs_parameter_t param;
	lbfgs_parameter_init(&param);
	param.max_iterations = maxIter;
	param.m = 20;

	// wrap additional arguments
	ParamsLBFGS instance;
	instance.first = this;
	instance.second.first = &input;
	instance.second.second = &output;

	// start LBFGS optimization
//	lbfgs(numParams, x, 0, &evaluateLBFGS, 0, &instance, &param);

	checkLBFGS(numParams, &instance, x);

	// copy optimized parameters back
//	W = Map<Matrix<lbfgsfloatval_t, Dynamic, Dynamic> >(x, W.rows(), W.cols());

	// free memory used by LBFGS
	lbfgs_free(x);

	return true;
}



MatrixXd MCGSM::sample(const MatrixXd& input) {
	return MatrixXd::Random(mDimOut, input.cols());
}



Array<double, 1, Dynamic> MCGSM::samplePosterior(const MatrixXd& input, const MatrixXd& output) {
	return Array<double, 1, Dynamic>::Random(1, input.cols());
}



ArrayXXd MCGSM::posterior(const MatrixXd& input, const MatrixXd& output) {
	return ArrayXXd::Random(mNumComponents, input.cols());
}



Array<double, 1, Dynamic> MCGSM::logLikelihood(const MatrixXd& input, const MatrixXd& output) {
	return ArrayXXd::Random(1, input.cols());
}
