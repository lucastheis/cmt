#ifndef PRECONDITIONER_H
#define PRECONDITIONER_H

#include "transform.h"
using MCM::Transform;

#include <map>
using std::pair;
using std::make_pair;

#include "Eigen/Core"
using Eigen::ArrayXXd;

namespace MCM {
	class Preconditioner {
		public:
			Preconditioner(const Transform&);
			virtual ~Preconditioner() { }

			virtual int dimIn() const;
			virtual int dimOut() const;

			virtual pair<ArrayXXd, ArrayXXd> operator()(const ArrayXXd& input, const ArrayXXd& output) const;
			virtual pair<ArrayXXd, ArrayXXd> inverse(const ArrayXXd& input, const ArrayXXd& output) const;

			// TODO:
//			virtual ArrayXXd logJacobian(const ArrayXXd& input, const ArrayXXd& output) const = 0;

		protected:
			Preconditioner();

		private:
			const Transform& mTransform;
	};
}

#endif
