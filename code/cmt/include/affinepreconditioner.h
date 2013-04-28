#ifndef AFFINEPRECONDITIONER_H
#define AFFINEPRECONDITIONER_H

#include "preconditioner.h"

#include "Eigen/Core"
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace CMT {
	class AffinePreconditioner : public Preconditioner {
		public:
			AffinePreconditioner(
				const VectorXd& meanIn,
				const VectorXd& meanOut,
				const MatrixXd& preIn,
				const MatrixXd& preOut,
				const MatrixXd& predictor);
			AffinePreconditioner(
				const VectorXd& meanIn,
				const VectorXd& meanOut,
				const MatrixXd& preIn,
				const MatrixXd& preInInv,
				const MatrixXd& preOut,
				const MatrixXd& preOutInv,
				const MatrixXd& predictor);
			AffinePreconditioner(const AffinePreconditioner& preconditioner);
			virtual ~AffinePreconditioner();

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
			inline MatrixXd preIn() const;
			inline MatrixXd preInInv() const;
			inline MatrixXd preOut() const;
			inline MatrixXd preOutInv() const;
			inline MatrixXd predictor() const;

		protected:
			VectorXd mMeanIn;
			VectorXd mMeanOut;
			MatrixXd mPreIn;
			MatrixXd mPreInInv;
			MatrixXd mPreOut;
			MatrixXd mPreOutInv;
			MatrixXd mPredictor;
			double mLogJacobian;
			MatrixXd mGradTransform;

			AffinePreconditioner();
	};
}



VectorXd CMT::AffinePreconditioner::meanIn() const {
	return mMeanIn;
}



VectorXd CMT::AffinePreconditioner::meanOut() const {
	return mMeanOut;
}



MatrixXd CMT::AffinePreconditioner::preIn() const {
	return mPreIn;
}



MatrixXd CMT::AffinePreconditioner::preInInv() const {
	return mPreInInv;
}



MatrixXd CMT::AffinePreconditioner::preOut() const {
	return mPreOut;
}



MatrixXd CMT::AffinePreconditioner::preOutInv() const {
	return mPreOutInv;
}



MatrixXd CMT::AffinePreconditioner::predictor() const {
	return mPredictor;
}

#endif
