#include "regularizer.h"
#include "utils.h"

#include "Eigen/Core"
using Eigen::MatrixXd;

CMT::Regularizer::Regularizer(double lambda, Norm norm) :
	mUseMatrix(false),
	mNorm(norm),
	mLambda(lambda)
{
}



CMT::Regularizer::Regularizer(MatrixXd matrix, Norm norm, double lambda) :
	mUseMatrix(true),
	mNorm(norm),
	mMatrix(matrix),
	mLambda(lambda)
{
	if(norm == L2)
		mMatrixMatrix = mMatrix.transpose() * mMatrix;
}



double CMT::Regularizer::evaluate(const MatrixXd& parameters) {
	if(mUseMatrix) {
		switch(mNorm) {
			case L1:
				return mLambda * (mMatrix * parameters).array().abs().sum();

			case L2:
				return mLambda * (mMatrix * parameters).array().square().sum();
		}
	} else {
		switch(mNorm) {
			case L1:
				return mLambda * parameters.array().abs().sum();

			case L2:
				return mLambda * parameters.array().square().sum();
		}
	}

	return 0.;
}



MatrixXd CMT::Regularizer::gradient(const MatrixXd& parameters) {
	if(mUseMatrix) {
		switch(mNorm) {
			case L1:
				return mLambda * mMatrix.transpose() * signum(mMatrix * parameters);

			case L2:
				return mLambda * 2. * mMatrixMatrix * parameters;
				
		}
	} else {
		switch(mNorm) {
			case L1:
				return mLambda * signum(parameters);

			case L2:
				return mLambda * 2. * parameters;
		}
	}

	return MatrixXd::Zero(parameters.rows(), parameters.cols());
}
