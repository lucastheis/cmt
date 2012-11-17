#ifndef TRANSFORM_H
#define TRANSFORM_H

#include "Eigen/Core"
using Eigen::ArrayXXd;
using Eigen::MatrixXd;

namespace MCM {
	class Transform {
		public:
			virtual ~Transform() { }

			virtual ArrayXXd operator()(const ArrayXXd& input) const = 0;
			virtual ArrayXXd inverse(const ArrayXXd& output) const = 0;
//			virtual ArrayXXd logJacobian(ArrayXXd input) const = 0;
	};
}

#endif
