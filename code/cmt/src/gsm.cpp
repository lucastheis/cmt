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

CMT::GSM::Parameters::Parameters() :
	maxIter(10)
{
}



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

	MatrixXd cov = CMT::covariance(MatrixXd::Random(dim, dim * dim));
	mCholesky.compute(cov.array() / pow(cov.determinant(), 1. / dim));
}



CMT::GSM* CMT::GSM::copy() {
	return new GSM(*this);
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

	ArrayXXd samples = mCholesky.matrixL() * sampleNormal(mDim, numSamples).matrix();
	return (samples.rowwise() / scales).colwise() + mMean.array();
}



Array<double, 1, Dynamic> CMT::GSM::logLikelihood(const MatrixXd& data) const {
	MatrixXd dataCentered = data.colwise() - mMean;
	Matrix<double, 1, Dynamic> sqNorm =
		(mCholesky.solve(dataCentered).array() * dataCentered.array()).colwise().sum();

	// normalization constant
	double logDet = MatrixXd(mCholesky.matrixL()).diagonal().array().log().sum();
	double logPtf = mDim / 2. * log(2. * PI) + logDet;

	// joint distribution over data and scales
	ArrayXXd logJoint = (-0.5 * mScales * sqNorm).array().colwise()
		+ (mPriors.array().log() + mDim / 2. * mScales.array().log() - logPtf);

	// marginalize out scales
	return logSumExp(logJoint);
}



bool CMT::GSM::train(const MatrixXd& data, const Parameters& parameters) {
	if(data.rows() != dim())
		throw Exception("Data has wrong dimensionality.");
	if(data.cols() == 0)
		return true;

	// compute mean
	mMean = data.rowwise().mean();

	// compute covariance
	MatrixXd dataCentered = data.colwise() - mMean;
	MatrixXd cov = dataCentered * dataCentered.transpose() / data.cols();
	mCholesky.compute(cov.array() / pow(cov.determinant(), 1. / mDim));

	// squared norm of whitened data
	Matrix<double, 1, Dynamic> sqNorm =
		(mCholesky.solve(dataCentered).array() * dataCentered.array()).colwise().sum();

	for(int i = 0; i < parameters.maxIter; ++i) {
		// unnormalized joint distribution over data and scales
		ArrayXXd logJoint = (-0.5 * mScales * sqNorm).array().colwise()
			+ (mPriors.array().log() + mDim / 2. * mScales.array().log());

		// compute posterior and responsibilities (E)
		Array<double, 1, Dynamic> uLogLik = logSumExp(logJoint);
		ArrayXXd post = (logJoint.rowwise() - uLogLik).exp();
		ArrayXd postSum = post.rowwise().sum();
		ArrayXXd weights = post.colwise() / postSum;

		// update priors and precisions (M)
		mPriors = postSum / data.cols();
		mScales = mDim / (weights.rowwise() * sqNorm.array()).rowwise().sum();
	}

	return true;
}



bool CMT::GSM::train(
	const MatrixXd& data,
	const Array<double, 1, Dynamic>& weights,
	const Parameters& parameters)
{
	// compute mean
	mMean = (data.array().rowwise() * weights).rowwise().sum();

	// compute covariance
	MatrixXd dataCentered = data.colwise() - mMean;
	MatrixXd dataWeighted = dataCentered.array().rowwise() * weights.sqrt();
	MatrixXd cov = dataWeighted * dataWeighted.transpose();
	mCholesky.compute(cov.array() / pow(cov.determinant(), 1. / mDim));

	// squared norm of whitened data
	Matrix<double, 1, Dynamic> sqNorm =
		(mCholesky.solve(dataCentered).array() * dataCentered.array()).colwise().sum();

	for(int i = 0; i < parameters.maxIter; ++i) {
		// unnormalized joint distribution over data and scales
		ArrayXXd logJoint = (-0.5 * mScales * sqNorm).array().colwise()
			+ (mPriors.array().log() + mDim / 2. * mScales.array().log());

		// compute weighted posterior and responsibilities (E)
		Array<double, 1, Dynamic> uLogLik = logSumExp(logJoint);
		ArrayXXd postWeighted = (logJoint.rowwise() - uLogLik).exp().rowwise() * weights;
		ArrayXd postWeightedSum = postWeighted.rowwise().sum();
		ArrayXXd scaleWeights = postWeighted.colwise() / postWeightedSum;

		// update priors and precisions (M)
		mPriors = postWeightedSum;
		mScales = mDim / (scaleWeights.rowwise() * sqNorm.array()).rowwise().sum();
	}

	return true;
}
