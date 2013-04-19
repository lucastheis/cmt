#include "utils.h"
#include "exception.h"
#include "whiteningtransform.h"

#include "Eigen/Eigenvalues"
using Eigen::SelfAdjointEigenSolver;

CMT::WhiteningTransform::WhiteningTransform(const ArrayXXd& input, const ArrayXXd& output) {
	if(input.cols() < input.rows())
		throw Exception("Too few inputs to compute whitening transform."); 

	mMeanIn = input.rowwise().mean();

	// compute covariances
	MatrixXd covXX = covariance(input);

	// input whitening
	SelfAdjointEigenSolver<MatrixXd> eigenSolver;

	eigenSolver.compute(covXX);
	mPreIn = eigenSolver.operatorInverseSqrt();
	mPreInInv = eigenSolver.operatorSqrt();

	mMeanOut = VectorXd::Zero(output.rows());
	mPreOut = MatrixXd::Identity(output.rows(), output.rows());
	mPreOutInv = MatrixXd::Identity(output.rows(), output.rows());
	mPredictor = MatrixXd::Zero(output.rows(), input.rows());
	mGradTransform = MatrixXd::Zero(output.rows(), input.rows());
	mLogJacobian = 1.;
}



CMT::WhiteningTransform::WhiteningTransform(const ArrayXXd& input, int dimOut) {
	if(input.cols() < input.rows())
		throw Exception("Too few inputs to compute whitening transform."); 

	mMeanIn = input.rowwise().mean();

	// compute covariances
	MatrixXd covXX = covariance(input);

	// input whitening
	SelfAdjointEigenSolver<MatrixXd> eigenSolver;

	eigenSolver.compute(covXX);
	mPreIn = eigenSolver.operatorInverseSqrt();
	mPreInInv = eigenSolver.operatorSqrt();

	mMeanOut = VectorXd::Zero(dimOut);
	mPreOut = MatrixXd::Identity(dimOut, dimOut);
	mPreOutInv = MatrixXd::Identity(dimOut, dimOut);
	mPredictor = MatrixXd::Zero(dimOut, input.rows());
	mGradTransform = MatrixXd::Zero(dimOut, input.rows());
	mLogJacobian = 1.;
}



CMT::WhiteningTransform::WhiteningTransform(
	const VectorXd& meanIn,
	const MatrixXd& preIn,
	const MatrixXd& preInInv,
	int dimOut) : AffineTransform(meanIn, preIn, preInInv, dimOut)
{
}
