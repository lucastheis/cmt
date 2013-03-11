#include "utils.h"
#include "exception.h"
#include "whiteningpreconditioner.h"

#include "Eigen/Eigenvalues"
using Eigen::SelfAdjointEigenSolver;

#include <iostream>

CMT::WhiteningPreconditioner::WhiteningPreconditioner(const ArrayXXd& input, const ArrayXXd& output) {
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

	// input whitening
	eigenSolver.compute(covXX);
	mWhiteIn = eigenSolver.operatorInverseSqrt();
	mWhiteInInv = eigenSolver.operatorSqrt();

	// optimal linear predictor
	mPredictor = covYX * mWhiteIn;

	// output whitening
	eigenSolver.compute(covYY - mPredictor * mPredictor.transpose());
	mWhiteOut = eigenSolver.operatorInverseSqrt();
	mWhiteOutInv = eigenSolver.operatorSqrt();

	// log-Jacobian determinant
	mLogJacobian = mWhiteOut.partialPivLu().matrixLU().diagonal().array().abs().log().sum();
}



CMT::WhiteningPreconditioner::WhiteningPreconditioner(
	const VectorXd& meanIn,
	const VectorXd& meanOut,
	const MatrixXd& whiteIn,
	const MatrixXd& whiteInInv,
	const MatrixXd& whiteOut,
	const MatrixXd& whiteOutInv,
	const MatrixXd& predictor) :
	mMeanIn(meanIn),
	mMeanOut(meanOut),
	mWhiteIn(whiteIn),
	mWhiteInInv(whiteInInv),
	mWhiteOut(whiteOut),
	mWhiteOutInv(whiteOutInv),
	mPredictor(predictor),
	mLogJacobian(mWhiteOut.partialPivLu().matrixLU().diagonal().array().abs().log().sum())
{
}



CMT::WhiteningPreconditioner::WhiteningPreconditioner() {
}



int CMT::WhiteningPreconditioner::dimIn() const {
	return mMeanIn.size();
}



int CMT::WhiteningPreconditioner::dimInPre() const {
	return mWhiteIn.rows();
}



int CMT::WhiteningPreconditioner::dimOut() const {
	return mMeanOut.size();
}



int CMT::WhiteningPreconditioner::dimOutPre() const {
	return mWhiteOut.rows();
}



pair<ArrayXXd, ArrayXXd> CMT::WhiteningPreconditioner::operator()(
	const ArrayXXd& input,
	const ArrayXXd& output) const
{
	if(input.cols() != output.cols())
		throw Exception("Number of inputs and outputs must be the same."); 
	if(input.rows() != dimIn())
		throw Exception("Input has wrong dimensionality."); 
	if(output.rows() != dimOut())
		throw Exception("Output has wrong dimensionality."); 
	ArrayXXd inputTr = mWhiteIn * (input.matrix().colwise() - mMeanIn);
	ArrayXXd outputTr = mWhiteOut * (output.matrix().colwise() - mMeanOut - mPredictor * inputTr.matrix());
	return make_pair(inputTr, outputTr);
}



pair<ArrayXXd, ArrayXXd> CMT::WhiteningPreconditioner::inverse(
	const ArrayXXd& input,
	const ArrayXXd& output) const
{
	if(input.cols() != output.cols())
		throw Exception("Number of inputs and outputs must be the same."); 
	if(input.rows() != dimInPre())
		throw Exception("Input has wrong dimensionality."); 
	if(output.rows() != dimOutPre())
		throw Exception("Output has wrong dimensionality."); 
	ArrayXXd outputTr = (mWhiteOutInv * output.matrix() + mPredictor * input.matrix()).colwise() + mMeanOut;
	ArrayXXd inputTr = (mWhiteInInv * input.matrix()).colwise() + mMeanIn;
	return make_pair(inputTr, outputTr);
}



ArrayXXd CMT::WhiteningPreconditioner::operator()(const ArrayXXd& input) const {
	if(input.rows() != dimIn())
		throw Exception("Input has wrong dimensionality."); 
	return mWhiteIn * (input.matrix().colwise() - mMeanIn);
}



ArrayXXd CMT::WhiteningPreconditioner::inverse(const ArrayXXd& input) const {
	if(input.rows() != dimInPre())
		throw Exception("Input has wrong dimensionality."); 
	return (mWhiteInInv * input.matrix()).colwise() + mMeanIn;
}



Array<double, 1, Dynamic> CMT::WhiteningPreconditioner::logJacobian(const ArrayXXd& input, const ArrayXXd& output) const {
	return Array<double, 1, Dynamic>::Zero(output.cols()) + mLogJacobian;
}
