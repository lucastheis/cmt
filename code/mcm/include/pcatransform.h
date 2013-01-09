#ifndef PCATRANSFORM_H
#define PCATRANSFORM_H

#include "affinetransform.h"

#include "Eigen/Core"
using Eigen::VectorXd;

namespace MCM {
	class PCATransform : public AffineTransform {
		public:
			PCATransform(const ArrayXXd& data, int numPCs = 0); 

			inline VectorXd eigenvalues() const;

		protected:
			VectorXd mEigenvalues;
	};
}



VectorXd MCM::PCATransform::eigenvalues() const {
	return mEigenvalues;
}

#endif
