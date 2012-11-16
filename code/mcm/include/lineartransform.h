#ifndef LINEARTRANSFORM_H
#define LINEARTRANSFORM_H

#include "transform.h"

namespace MCM {
	class LinearTransform : public Transform {
		public:
			LinearTransform(MatrixXd mat);

			virtual ArrayXXd operator()(ArrayXXd input) const;
			virtual ArrayXXd inverse(ArrayXXd output) const;

		protected:
			MatrixXd mMat;
			mutable MatrixXd mMatInverse;
	};
}

#endif
