#ifndef LINEARTRANSFORM_H
#define LINEARTRANSFORM_H

#include "transform.h"

namespace MCM {
	class LinearTransform : public Transform {
		public:
			LinearTransform(MatrixXd mat);

			inline MatrixXd matrix() const;
			inline void setMatrix(MatrixXd mat);

			virtual ArrayXXd operator()(ArrayXXd input) const;
			virtual ArrayXXd inverse(ArrayXXd output) const;

		protected:
			MatrixXd mMat;
			mutable MatrixXd mMatInverse;
	};
}



MatrixXd MCM::LinearTransform::matrix() const {
	return mMat;
}



void MCM::LinearTransform::setMatrix(MatrixXd mat) {
	mMat = mat;
	mMatInverse = MatrixXd();
}

#endif
