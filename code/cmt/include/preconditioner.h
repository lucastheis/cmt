#ifndef PRECONDITIONER_H
#define PRECONDITIONER_H

#include <map>
using std::pair;
using std::make_pair;

#include "Eigen/Core"
using Eigen::ArrayXXd;

namespace CMT {
	class Preconditioner {
		public:
			virtual ~Preconditioner() { }

			virtual int dimIn() const = 0;
			virtual int dimInPre() const = 0;
			virtual int dimOut() const = 0;
			virtual int dimOutPre() const = 0;

			virtual pair<ArrayXXd, ArrayXXd> operator()(const ArrayXXd& input, const ArrayXXd& output) const = 0;
			virtual pair<ArrayXXd, ArrayXXd> inverse(const ArrayXXd& input, const ArrayXXd& output) const = 0;

			virtual ArrayXXd operator()(const ArrayXXd& input) const = 0;
			virtual ArrayXXd inverse(const ArrayXXd& input) const = 0;

			virtual Array<double, 1, Dynamic> logJacobian(const ArrayXXd& input, const ArrayXXd& output) const = 0;

			virtual pair<ArrayXXd, ArrayXXd> adjustGradient(
				const ArrayXXd& inputGradient,
				const ArrayXXd& outputGradient) const = 0;
	};
}

#endif
