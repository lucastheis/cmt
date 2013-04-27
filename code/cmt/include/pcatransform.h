#ifndef PCATRANSFORM_H
#define PCATRANSFORM_H

#include "affinetransform.h"

namespace CMT {
	class PCATransform : public AffineTransform {
		public:
			PCATransform(
				const ArrayXXd& input,
				const ArrayXXd& output,
				double varExplained = 99.,
				int numPCs = -1);
			PCATransform(
				const ArrayXXd& input,
				double varExplained = 99.,
				int numPCs = -1,
				int dimOut = 1);
			PCATransform(
				const VectorXd& eigenvalues,
				const VectorXd& meanIn,
				const MatrixXd& preIn,
				const MatrixXd& preInInv,
				int dimOut);

			inline VectorXd eigenvalues() const;

		protected:
			VectorXd mEigenvalues;

		private:
			void initialize(const ArrayXXd& input, double varExplained, int numPCs, int dimOut);
	};
}



VectorXd CMT::PCATransform::eigenvalues() const {
	return mEigenvalues;
}

#endif
