#include "regularizer.h"
#include "utils.h"

#include "Eigen/Core"
using Eigen::MatrixXd;

CMT::Regularizer::Regularizer(double strength, Norm norm) :
	mUseMatrix(false),
	mNorm(norm),
	mStrength(strength)
{
}



CMT::Regularizer::Regularizer(MatrixXd matrix, Norm norm, double strength) :
	mUseMatrix(true),
	mNorm(norm),
	mStrength(strength),
	mTransform(matrix)
{
	if(norm == L2)
		mTT = mTransform.transpose() * mTransform;
}



double CMT::Regularizer::evaluate(const MatrixXd& parameters) const {
	if(mUseMatrix) {
		if(mTransform.cols() != parameters.rows())
			throw Exception("Regularizer transform and parameters are incompatible.");

		switch(mNorm) {
			case L1:
				return mStrength * (mTransform * parameters).array().abs().sum();

			case L2:
				return mStrength * (mTransform * parameters).array().square().sum();
		}
	} else {
		switch(mNorm) {
			case L1:
				return mStrength * parameters.array().abs().sum();

			case L2:
				return mStrength * parameters.array().square().sum();
		}
	}

	return 0.;
}



MatrixXd CMT::Regularizer::gradient(const MatrixXd& parameters) const {
	if(mUseMatrix) {
		if(mTransform.cols() != parameters.rows())
			throw Exception("Regularizer transform and parameters are incompatible.");

		switch(mNorm) {
			case L1:
				return mStrength * mTransform.transpose() * signum(mTransform * parameters);

			case L2:
				return mStrength * 2. * mTT * parameters;
				
		}
	} else {
		switch(mNorm) {
			case L1:
				return mStrength * signum(parameters);

			case L2:
				return mStrength * 2. * parameters;
		}
	}

	return MatrixXd::Zero(parameters.rows(), parameters.cols());
}
