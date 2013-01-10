#ifndef PRECONDITIONER_H
#define PRECONDITIONER_H

#include <map>
using std::pair;
using std::make_pair;

#include "Eigen/Core"
using Eigen::ArrayXXd;

namespace MCM {
	class Preconditioner {
		public:
			virtual ~Preconditioner() { }

			virtual int dimIn() const = 0;
			virtual int dimOut() const = 0;

			virtual pair<ArrayXXd, ArrayXXd> operator()(const ArrayXXd& input, const ArrayXXd& output) const = 0;
			virtual pair<ArrayXXd, ArrayXXd> inverse(const ArrayXXd& input, const ArrayXXd& output) const = 0;

			// TODO:
//			virtual ArrayXXd logJacobian(const ArrayXXd& input, const ArrayXXd& output) const = 0;
	};
}

#endif
