#ifndef TRANSFORM_H
#define TRANSFORM_H

#include "Eigen/Core"

namespace MCM {
	class Transform {
		public:
			virtual ~Transform() { }

			virtual ArrayXXd operator()(ArrayXXd input) const = 0;
			virtual ArrayXXd inverse(ArrayXXd output) const = 0;
//			virtual ArrayXXd logJacobian(ArrayXXd input) const = 0;
	};
}

#endif
