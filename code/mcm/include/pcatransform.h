#ifndef PCATRANSFORM_H
#define PCATRANSFORM_H

#include "Eigen/Core"
using Eigen::VectorXd;

#include "lineartransform.h"

namespace MCM {
	class PCATransform : public LinearTransform {
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
