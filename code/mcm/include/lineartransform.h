#ifndef LINEARTRANSFORM_H
#define LINEARTRANSFORM_H

#include "affinetransform.h"

namespace MCM {
	class LinearTransform : public AffineTransform {
		public:
			LinearTransform(const MatrixXd& mat);

			virtual ArrayXXd operator()(const ArrayXXd& data) const;
			virtual ArrayXXd inverse(const ArrayXXd& data) const;
	};
}

#endif
