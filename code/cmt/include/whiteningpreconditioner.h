#ifndef WHITENINGPRECONDITIONER_H
#define WHITENINGPRECONDITIONER_H

#include "preconditioner.h"

#include "Eigen/Core"
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace CMT {
	class WhiteningPreconditioner : public Preconditioner {
		public:
			WhiteningPreconditioner(const ArrayXXd& input, const ArrayXXd& output);
			WhiteningPreconditioner(
				const VectorXd& meanIn,
				const VectorXd& meanOut,
				const MatrixXd& whiteIn,
				const MatrixXd& whiteInInv,
				const MatrixXd& whiteOut,
				const MatrixXd& whiteOutInv,
				const MatrixXd& predictor);

			virtual int dimIn() const;
			virtual int dimInPre() const;
			virtual int dimOut() const;
			virtual int dimOutPre() const;

			virtual pair<ArrayXXd, ArrayXXd> operator()(const ArrayXXd& input, const ArrayXXd& output) const;
			virtual pair<ArrayXXd, ArrayXXd> inverse(const ArrayXXd& input, const ArrayXXd& output) const;

			virtual ArrayXXd operator()(const ArrayXXd& input) const;
			virtual ArrayXXd inverse(const ArrayXXd& input) const;

			virtual Array<double, 1, Dynamic> logJacobian(const ArrayXXd& input, const ArrayXXd& output) const;

			virtual pair<ArrayXXd, ArrayXXd> adjustGradient(
				const ArrayXXd& inputGradient,
				const ArrayXXd& outputGradient) const;

			inline VectorXd meanIn() const;
			inline VectorXd meanOut() const;
			inline MatrixXd whiteIn() const;
			inline MatrixXd whiteInInv() const;
			inline MatrixXd whiteOut() const;
			inline MatrixXd whiteOutInv() const;
			inline MatrixXd predictor() const;

		protected:
			VectorXd mMeanIn;
			VectorXd mMeanOut;
			MatrixXd mWhiteIn;
			MatrixXd mWhiteInInv;
			MatrixXd mWhiteOut;
			MatrixXd mWhiteOutInv;
			MatrixXd mPredictor;
			double mLogJacobian;
			MatrixXd mGradTransform;

			WhiteningPreconditioner();
	};
}



VectorXd CMT::WhiteningPreconditioner::meanIn() const {
	return mMeanIn;
}



VectorXd CMT::WhiteningPreconditioner::meanOut() const {
	return mMeanOut;
}



MatrixXd CMT::WhiteningPreconditioner::whiteIn() const {
	return mWhiteIn;
}



MatrixXd CMT::WhiteningPreconditioner::whiteInInv() const {
	return mWhiteInInv;
}



MatrixXd CMT::WhiteningPreconditioner::whiteOut() const {
	return mWhiteOut;
}



MatrixXd CMT::WhiteningPreconditioner::whiteOutInv() const {
	return mWhiteOutInv;
}



MatrixXd CMT::WhiteningPreconditioner::predictor() const {
	return mPredictor;
}

#endif
