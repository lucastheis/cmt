#include "mcgsm.h"
#include "utils.h"
#include "lbfgs.h"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <utility>

using namespace std;

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
	mNumFeatures(numFeatures)
{
	// initialize parameters
	mPriors = ArrayXXd::Random(mNumComponents, mNumScales).abs() / 5. + 0.9;
	mScales = ArrayXXd::Random(mNumComponents, mNumScales).abs() / 2. + 0.75;
	mWeights = ArrayXXd::Random(mNumComponents, mNumFeatures).abs() / 10. + 0.01;
	mFeatures = ArrayXXd::Random(mDimIn, mNumFeatures) / 10.;

	for(int i = 0; i < mNumComponents; ++i) {
		mCholeskyFactors.push_back(VectorXd::Ones(mDimOut).asDiagonal());
		mPredictors.push_back(ArrayXXd::Random(mDimOut, mDimIn) / 10.);
	}
}



void MCGSM::normalize() {
}



bool MCGSM::train(const MatrixXd& input, const MatrixXd& output, int maxIter, double tol) {
	if(input.rows() != mDimIn || output.rows() != mDimOut)
		throw Exception("Data has wrong dimensionality.");

	return false;
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
