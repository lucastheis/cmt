#ifndef CMT_NONLINEARITIES_H
#define CMT_NONLINEARITIES_H

#include "Eigen/Core"

namespace CMT {
	using Eigen::ArrayXXd;

	class Nonlinearity {
		public:
			virtual ~Nonlinearity();
			virtual ArrayXXd operator()(const ArrayXXd& data) const = 0;
			virtual ArrayXXd derivative(const ArrayXXd& data) const = 0;
	};

	class LogisticFunction : public Nonlinearity {
		virtual ArrayXXd operator()(const ArrayXXd& data) const;
		virtual ArrayXXd derivative(const ArrayXXd& data) const;
	};

}

#endif
