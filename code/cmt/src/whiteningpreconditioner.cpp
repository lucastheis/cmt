#include "utils.h"
#include "exception.h"
#include "whiteningpreconditioner.h"

#include "Eigen/Eigenvalues"
using Eigen::SelfAdjointEigenSolver;

CMT::WhiteningPreconditioner::WhiteningPreconditioner(const ArrayXXd& input, const ArrayXXd& output) {
	if(input.rows() == 0) {
		if(output.cols() < output.rows())
			throw Exception("Too few inputs to compute whitening transform."); 

		mMeanOut = output.rowwise().mean();

		MatrixXd cov = covariance(output);

		SelfAdjointEigenSolver<MatrixXd> eigenSolver;

		// output whitening
		eigenSolver.compute(cov);
		mPreOut = eigenSolver.operatorInverseSqrt();
		mPreOutInv = eigenSolver.operatorSqrt();

		// log-Jacobian determinant
		mLogJacobian = mPreOut.partialPivLu().matrixLU().diagonal().array().abs().log().sum();
	} else {
		if(input.cols() != output.cols())
			throw Exception("Number of inputs and outputs must be the same."); 
		if(input.cols() < input.rows())
			throw Exception("Too few inputs to compute whitening transform."); 

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
		mPreIn = eigenSolver.operatorInverseSqrt();
		mPreInInv = eigenSolver.operatorSqrt();

		// optimal linear predictor
		mPredictor = covYX * mPreIn;

		// output whitening
		eigenSolver.compute(covYY - mPredictor * mPredictor.transpose());
		mPreOut = eigenSolver.operatorInverseSqrt();
		mPreOutInv = eigenSolver.operatorSqrt();

		// log-Jacobian determinant
		mLogJacobian = mPreOut.partialPivLu().matrixLU().diagonal().array().abs().log().sum();

		// used for transforming gradients
		mGradTransform = mPreOut * mPredictor * mPreIn;
	}
}



CMT::WhiteningPreconditioner::WhiteningPreconditioner(
	const VectorXd& meanIn,
	const VectorXd& meanOut,
	const MatrixXd& preIn,
	const MatrixXd& preInInv,
	const MatrixXd& preOut,
	const MatrixXd& preOutInv,
	const MatrixXd& predictor) :
	AffinePreconditioner(
		meanIn, meanOut, preIn, preInInv, preOut, preOutInv, predictor)
{
}
