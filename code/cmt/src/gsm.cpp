#include "gsm.h"
#include "utils.h"
#include "Eigen/LU"

#include "Eigen/Core"
using Eigen::Array;
using Eigen::ArrayXd;
using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::Dynamic;

#include <cmath>
using std::sqrt;

#include <cstdlib>
using std::rand;

CMT::GSM::GSM(int dim, int numScales) :
	mDim(dim),
	mMean(VectorXd::Zero(dim)),
	mPriors(VectorXd::Ones(numScales) / numScales),
	mScales(ArrayXd::Random(numScales).abs() / 2. + 0.75)
{
	if(dim < 1)
		throw Exception("Dimensionality has to be positive.");
	if(numScales < 1)
		throw Exception("Number of scales has to be positive.");

	setCovariance(CMT::covariance(MatrixXd::Random(dim, dim * dim)));
}



CMT::GSM* CMT::GSM::copy() {
	return new GSM(*this);
}



CMT::Mixture::Component& CMT::GSM::operator=(const Mixture::Component& component) {
	// requires that component is a GSM
	const GSM& gsm = dynamic_cast<const GSM&>(component);

	mDim = gsm.mDim;
	mMean = gsm.mMean;
	mScales = gsm.mScales;
	mCholesky = gsm.mCholesky;
}



MatrixXd CMT::GSM::sample(int numSamples) const {
	Array<double, 1, Dynamic> scales(numSamples);

	// cumulative distribution function
	ArrayXd cdf = mPriors;
	for(int k = 1; k < numScales(); ++k)
		cdf[k] += cdf[k - 1];

	// make sure last entry is definitely large enough
	cdf[numScales() - 1] = 1.0001;

	// sample scales
	for(int i = 0; i < numSamples; ++i) {
		double urand = static_cast<double>(rand()) / (static_cast<long>(RAND_MAX) + 1l);

		int j = 0;
		while(urand > cdf[j])
			++j;
		scales[i] = sqrt(mScales[j]);
	}

	MatrixXd samples = mCholesky.triangularView<Eigen::Lower>().solve(sampleNormal(mDim, numSamples).matrix());
	return (samples.array().rowwise() / scales).colwise() + mMean.array();
}



Array<double, 1, Dynamic> CMT::GSM::logLikelihood(const MatrixXd& data) const {
	MatrixXd dataWhitened = mCholesky * (data.colwise() - mMean);
	Matrix<double, 1, Dynamic> sqNorm = dataWhitened.colwise().squaredNorm();

	// normalization constant
	double logDet = mCholesky.diagonal().array().log().sum();
	double logPtf = mDim / 2. * log(2. * PI) - logDet;

	// joint distribution over data and scales
	ArrayXXd logJoint = (-0.5 * mScales * sqNorm).array().colwise()
		+ (mPriors.array().log() + mDim / 2. * mScales.array().log() - logPtf);

	// marginalize out scales
	return logSumExp(logJoint);
}



void CMT::GSM::initialize(const MatrixXd& data, const Parameters& parameters) {
	if(data.rows() != dim())
		throw Exception("Data has wrong dimensionality.");

	// data statistics
	VectorXd mean = data.rowwise().mean();
	MatrixXd cov = CMT::covariance(data)
		+ parameters.regularizeCovariance * MatrixXd::Identity(dim(), dim());
	MatrixXd cholesky = LLT<MatrixXd>(cov).matrixL();

	// draw samples with 
	MatrixXd samples = cholesky * sampleNormal(dim(), dim() * dim()).matrix();
	samples.colwise() += mean;

	if(parameters.trainCovariance)
		setCovariance(CMT::covariance(samples));

	if(parameters.trainMean)
		setMean(samples.rowwise().mean());

	if(parameters.trainScales)
		mScales = ArrayXd::Random(numScales()).abs() / 2. + 0.75;
}



bool CMT::GSM::train(const MatrixXd& data, const Parameters& parameters) {
	if(data.rows() != dim())
		throw Exception("Data has wrong dimensionality.");
	if(data.cols() == 0)
		return true;

	if(parameters.trainMean)
		// update mean
		mMean = data.rowwise().mean();

	MatrixXd dataCentered = data.colwise() - mMean;

	if(parameters.trainCovariance) {
		MatrixXd cov = dataCentered * dataCentered.transpose() / data.cols();

		// compute Cholesky factor of covariance matrix
		MatrixXd cholesky = LLT<MatrixXd>(cov).matrixL();

		// update Cholesky factor of precision matrix
		mCholesky = cholesky.triangularView<Eigen::Lower>().solve(
			MatrixXd::Identity(dim(), dim()));
	}

	// squared norm of whitened data
	Matrix<double, 1, Dynamic> sqNorm = (mCholesky * dataCentered).colwise().squaredNorm();

	for(int i = 0; i < parameters.maxIter; ++i) {
		// unnormalized joint distribution over data and scales
		ArrayXXd logJoint = (-0.5 * mScales * sqNorm).array().colwise()
			+ (mPriors.array().log() + mDim / 2. * mScales.array().log());

		// compute posterior and responsibilities (E)
		Array<double, 1, Dynamic> uLogLik = logSumExp(logJoint);
		ArrayXXd post = (logJoint.rowwise() - uLogLik).eval().exp();
		ArrayXd postSum = post.rowwise().sum();

		// update prior weights and precision scale variables (M)
		if(parameters.trainPriors) {
			mPriors = postSum / data.cols() + parameters.regularizePriors;
			mPriors /= mPriors.sum();
		}

		if(parameters.trainScales) {
			ArrayXXd weights = post.colwise() / postSum;
			mScales = mDim / (weights.rowwise() * sqNorm.array()).rowwise().sum();
		}
	}

	return true;
}



/**
 * Fits parameters to given weighted data using expectation maximization (EM).
 *
 * @param data data stored column-wise
 * @param weights weights corresponding to data points which sum to one
 * @param parameters hyperparameters which control the optimization and regularization
 */
bool CMT::GSM::train(
	const MatrixXd& data,
	const Array<double, 1, Dynamic>& weights,
	const Parameters& parameters)
{
	if(data.cols() == 0)
		return true;
	if(data.rows() != dim())
		throw Exception("Data has wrong dimensionality.");
	if(data.cols() != weights.cols())
		throw Exception("Wrong number of weights.");

	if(parameters.trainMean)
		// update mean
		mMean = (data.array().rowwise() * weights).rowwise().sum();

	MatrixXd dataCentered = data.colwise() - mMean;

	if(parameters.trainCovariance) {

		MatrixXd dataWeighted = dataCentered.array().rowwise() * weights.sqrt();
		MatrixXd cov = dataWeighted * dataWeighted.transpose();

		// compute Cholesky factor of covariance matrix
		MatrixXd cholesky = LLT<MatrixXd>(cov).matrixL();

		// update Cholesky factor of precision matrix
		mCholesky = cholesky.triangularView<Eigen::Lower>().solve(
			MatrixXd::Identity(dim(), dim()));
	}

	// squared norm of whitened data
	Matrix<double, 1, Dynamic> sqNorm = (mCholesky * dataCentered).colwise().squaredNorm();

	for(int i = 0; i < parameters.maxIter; ++i) {
		// unnormalized joint distribution over data and scales
		ArrayXXd logJoint = (-0.5 * mScales * sqNorm).array().colwise()
			+ (mPriors.array().log() + mDim / 2. * mScales.array().log());

		// compute posterior and responsibilities (E)
		ArrayXXd postWeighted = logJoint.rowwise() - logJoint.colwise().maxCoeff();
		postWeighted = postWeighted.exp();
		postWeighted.rowwise() *= weights / postWeighted.colwise().sum();

		ArrayXd postWeightedSum = postWeighted.rowwise().sum();

		// update prior weights and precision scale variables (M)
		if(parameters.trainPriors)
			mPriors = postWeightedSum;

		if(parameters.trainScales) {
			ArrayXXd scaleWeights = postWeighted.colwise() / postWeightedSum;
			mScales = mDim / (scaleWeights.rowwise() * sqNorm.array()).rowwise().sum();
		}
	}

	return true;
}
