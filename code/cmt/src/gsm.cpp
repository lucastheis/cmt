#include "gsm.h"
#include "utils.h"

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
	mPriors(VectorXd::Ones(numScales)),
	mScales(ArrayXd::Random(numScales).abs() / 2. + 0.75)
{
	MatrixXd cov = CMT::covariance(MatrixXd::Random(dim, dim * dim));
	mCholesky.compute(cov.array() / pow(cov.determinant(), 1. / dim));
}



MatrixXd CMT::GSM::sample(int numSamples) const {
	Array<double, 1, Dynamic> scales(numSamples);

	// cumulative distribution function
	ArrayXd cdf = mPriors;
	for(int k = 1; k < numScales(); ++k)
		cdf[k] += cdf[k - 1];

	// make sure last entry is large enough
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
	MatrixXd dataCentered = mCholesky.solve(data.colwise() - mMean);
	Matrix<double, 1, Dynamic> sqNorm = dataCentered.colwise().squaredNorm();

	// normalization constant
	double logDet = MatrixXd(mCholesky.matrixL()).diagonal().array().log().sum();
	double logPtf = mDim / 2. * log(2. * PI) + logDet;

	// joint distribution over data and scales
	ArrayXXd logJoint = (-0.5 * mScales * sqNorm).array().colwise()
		+ (mPriors.array().log() - mDim / 2. * mScales.array().log() - logPtf);

	// marginalize out scales
	return logSumExp(logJoint);
}



bool CMT::GSM::train(const MatrixXd& data) {
	return true;
}



bool CMT::GSM::train(
	const MatrixXd& data,
	const Array<double, 1, Dynamic>& weights)
{
	return true;
}
