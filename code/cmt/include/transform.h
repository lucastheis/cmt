#ifndef TRANSFORM_H
#define TRANSFORM_H

#include "Eigen/Core"
using Eigen::ArrayXXd;

namespace CMT {
	class Transform {
		public:
			virtual ~Transform() { }

			virtual int dim() const = 0;
			virtual int dimTr() const = 0;

			virtual ArrayXXd operator()(const ArrayXXd& data) const = 0;
			virtual ArrayXXd inverse(const ArrayXXd& data) const = 0;
	};
}

#endif
