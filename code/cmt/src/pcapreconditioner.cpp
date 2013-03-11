#include "utils.h"
#include "exception.h"
#include "pcapreconditioner.h"

#include "Eigen/Eigenvalues"
using Eigen::SelfAdjointEigenSolver;

#include <iostream>

CMT::PCAPreconditioner::PCAPreconditioner(
	const ArrayXXd& input,
	const ArrayXXd& output,
	double varExplained,
	int numPCs)
{
	if(input.cols() != output.cols())
		throw Exception("Number of inputs and outputs must be the same."); 

	mMeanIn = input.rowwise().mean();
	mMeanOut = output.rowwise().mean();

	// concatenate input and output
	ArrayXXd data(input.rows() + output.rows(), input.cols());
	data << input.matrix().colwise() - mMeanIn, output.matrix().colwise() - mMeanOut;

	// compute covariances
	MatrixXd cov = covariance(data);
	MatrixXd covXX = cov.topLeftCorner(input.rows(), input.rows());
	MatrixXd covYX = cov.bottomLeftCorner(output.rows(), input.rows());
	MatrixXd covYY = cov.bottomRightCorner(output.rows(), output.rows());

	SelfAdjointEigenSolver<MatrixXd> eigenSolver;
	eigenSolver.compute(covXX);
	mEigenvalues = eigenSolver.eigenvalues();

	if(numPCs < 0) {
		double totalVariance = mEigenvalues.sum();
		double varExplainedSoFar = 0.;
		numPCs = 0;

		for(int i = mEigenvalues.size() - 1; i >= 0; --i, ++numPCs) {
			varExplainedSoFar += mEigenvalues[i] / totalVariance;
			if(varExplainedSoFar > varExplained)
				break;
		}
	}

	// input whitening
	mWhiteIn = mEigenvalues.tail(numPCs).cwiseSqrt().cwiseInverse().asDiagonal() *
		eigenSolver.eigenvectors().rightCols(numPCs).transpose();
	mWhiteInInv = eigenSolver.eigenvectors().rightCols(numPCs) *
		mEigenvalues.tail(numPCs).cwiseSqrt().asDiagonal();

	// optimal linear predictor
	mPredictor = covYX * mWhiteIn.transpose();

	// output whitening
	eigenSolver.compute(covYY - mPredictor * mPredictor.transpose());
	mWhiteOut = eigenSolver.operatorInverseSqrt();
	mWhiteOutInv = eigenSolver.operatorSqrt();

	// log-Jacobian determinant
	mLogJacobian = mWhiteOut.partialPivLu().matrixLU().diagonal().array().abs().log().sum();
}



CMT::PCAPreconditioner::PCAPreconditioner(
	const VectorXd& eigenvalues,
	const VectorXd& meanIn,
	const VectorXd& meanOut,
	const MatrixXd& whiteIn,
	const MatrixXd& whiteInInv,
	const MatrixXd& whiteOut,
	const MatrixXd& whiteOutInv,
	const MatrixXd& predictor) :
	WhiteningPreconditioner(
		meanIn, meanOut, whiteIn, whiteInInv, whiteOut, whiteOutInv, predictor),
	mEigenvalues(eigenvalues)
{
}
