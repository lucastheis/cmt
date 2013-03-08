#ifndef WHITENINGPRECONDITIONER_H
#define WHITENINGPRECONDITIONER_H

#include "preconditioner.h"

#include "Eigen/Core"
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace MCM {
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
	};
}



VectorXd MCM::WhiteningPreconditioner::meanIn() const {
	return mMeanIn;
}



VectorXd MCM::WhiteningPreconditioner::meanOut() const {
	return mMeanOut;
}



MatrixXd MCM::WhiteningPreconditioner::whiteIn() const {
	return mWhiteIn;
}



MatrixXd MCM::WhiteningPreconditioner::whiteInInv() const {
	return mWhiteInInv;
}



MatrixXd MCM::WhiteningPreconditioner::whiteOut() const {
	return mWhiteOut;
}



MatrixXd MCM::WhiteningPreconditioner::whiteOutInv() const {
	return mWhiteOutInv;
}



MatrixXd MCM::WhiteningPreconditioner::predictor() const {
	return mPredictor;
}

#endif
