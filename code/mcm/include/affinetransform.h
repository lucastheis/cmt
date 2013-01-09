#ifndef AFFINETRANSFORM_H
#define AFFINETRANSFORM_H

#include "transform.h"

#include "Eigen/Core"
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace MCM {
	class AffineTransform : public Transform {
		public:
			AffineTransform(const MatrixXd& mat, const VectorXd& vec);

			inline MatrixXd matrix() const;
			inline void setMatrix(const MatrixXd& mat);

			inline VectorXd vector() const;
			inline void setVector(const VectorXd& vec);

			virtual int dimIn() const;
			virtual int dimOut() const;

			virtual ArrayXXd operator()(const ArrayXXd& data) const;
			virtual ArrayXXd inverse(const ArrayXXd& data) const;

		protected:
			MatrixXd mMat;
			VectorXd mVec;

			mutable MatrixXd mMatInverse;
	};
}



MatrixXd MCM::AffineTransform::matrix() const {
	return mMat;
}



void MCM::AffineTransform::setMatrix(const MatrixXd& mat) {
	mMat = mat;
	mMatInverse = MatrixXd();
}



VectorXd MCM::AffineTransform::vector() const {
	return mVec;
}



void MCM::AffineTransform::setVector(const VectorXd& vec) {
	mVec = vec;
}

#endif
