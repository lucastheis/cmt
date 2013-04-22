#include "utils.h"
#include "exception.h"
#include "pcatransform.h"

#include "Eigen/Eigenvalues"
using Eigen::SelfAdjointEigenSolver;

CMT::PCATransform::PCATransform(
	const ArrayXXd& input,
	const ArrayXXd& output,
	double varExplained,
	int numPCs)
{
	initialize(input, varExplained, numPCs, output.rows());
}



CMT::PCATransform::PCATransform(
	const ArrayXXd& input,
	double varExplained,
	int numPCs,
	int dimOut)
{
	initialize(input, varExplained, numPCs, dimOut);
}



void CMT::PCATransform::initialize(
	const ArrayXXd& input,
	double varExplained,
	int numPCs,
	int dimOut)
{
	mMeanIn = input.rowwise().mean();

	// compute covariances
	MatrixXd covXX = covariance(input);

	SelfAdjointEigenSolver<MatrixXd> eigenSolver;
	eigenSolver.compute(covXX);
	mEigenvalues = eigenSolver.eigenvalues();

	if(numPCs < 0) {
		double totalVariance = mEigenvalues.sum();
		double varExplainedSoFar = 0.;
		numPCs = 0;

		for(int i = mEigenvalues.size() - 1; i >= 0; --i, ++numPCs) {
			varExplainedSoFar += mEigenvalues[i] / totalVariance * 100.;
			if(varExplainedSoFar > varExplained)
				break;
		}
	} else if(numPCs > mEigenvalues.size()) {
		numPCs = mEigenvalues.size();
	}

	// input whitening
	mPreIn = mEigenvalues.tail(numPCs).cwiseSqrt().cwiseInverse().asDiagonal() *
		eigenSolver.eigenvectors().rightCols(numPCs).transpose();
	mPreInInv = eigenSolver.eigenvectors().rightCols(numPCs) *
		mEigenvalues.tail(numPCs).cwiseSqrt().asDiagonal();

	mMeanOut = VectorXd::Zero(dimOut);
	mPreOut = MatrixXd::Identity(dimOut, dimOut);
	mPreOutInv = MatrixXd::Identity(dimOut, dimOut);
	mPredictor = MatrixXd::Zero(dimOut, numPCs);
	mGradTransform = MatrixXd::Zero(dimOut, input.rows());
	mLogJacobian = 1.;
}



CMT::PCATransform::PCATransform(
	const VectorXd& eigenvalues,
	const VectorXd& meanIn,
	const MatrixXd& preIn,
	const MatrixXd& preInInv,
	int dimOut) :
	AffineTransform(meanIn, preIn, preInInv, dimOut),
	mEigenvalues(eigenvalues)
{
}
