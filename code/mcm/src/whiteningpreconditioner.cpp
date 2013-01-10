#include "utils.h"
#include "exception.h"
#include "whiteningpreconditioner.h"

#include "Eigen/Eigenvalues"
using Eigen::SelfAdjointEigenSolver;

MCM::WhiteningPreconditioner::WhiteningPreconditioner(const ArrayXXd& input, const ArrayXXd& output) {
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

	MatrixXd tmp = covYX * mWhiteIn;

	// output whitening
	eigenSolver.compute(covYY - tmp * tmp.transpose());
	mWhiteOut = eigenSolver.operatorInverseSqrt();
	mWhiteOutInv = eigenSolver.operatorSqrt();

	// output prediction
	mPredictor = covYX * mWhiteIn;
}



MCM::WhiteningPreconditioner::WhiteningPreconditioner(
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
	mPredictor(predictor)
{
}



int MCM::WhiteningPreconditioner::dimIn() const {
	return mPredictor.cols();
}



int MCM::WhiteningPreconditioner::dimOut() const {
	return mPredictor.rows();
}



pair<ArrayXXd, ArrayXXd> MCM::WhiteningPreconditioner::operator()(const ArrayXXd& input, const ArrayXXd& output) const {
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



pair<ArrayXXd, ArrayXXd> MCM::WhiteningPreconditioner::inverse(const ArrayXXd& input, const ArrayXXd& output) const {
	ArrayXXd outputTr = (mWhiteOutInv * output.matrix() + mPredictor * input.matrix()).colwise() + mMeanOut;
	ArrayXXd inputTr = (mWhiteInInv * input.matrix()).colwise() + mMeanIn;
	return make_pair(inputTr, outputTr);
}
