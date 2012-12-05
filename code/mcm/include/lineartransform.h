#ifndef LINEARTRANSFORM_H
#define LINEARTRANSFORM_H

#include "transform.h"

namespace MCM {
	class LinearTransform : public Transform {
		public:
			LinearTransform(const MatrixXd& mat);

			inline MatrixXd matrix() const;
			inline void setMatrix(const MatrixXd& mat);

			virtual int dimIn() const;
			virtual int dimOut() const;

			virtual ArrayXXd operator()(const ArrayXXd& input) const;
			virtual ArrayXXd inverse(const ArrayXXd& output) const;

		protected:
			MatrixXd mMat;
			mutable MatrixXd mMatInverse;
	};
}



MatrixXd MCM::LinearTransform::matrix() const {
	return mMat;
}



void MCM::LinearTransform::setMatrix(const MatrixXd& mat) {
	mMat = mat;
	mMatInverse = MatrixXd();
}

#endif
