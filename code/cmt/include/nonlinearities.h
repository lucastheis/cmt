#ifndef CMT_NONLINEARITIES_H
#define CMT_NONLINEARITIES_H

#include "Eigen/Core"

namespace CMT {
	using Eigen::ArrayXXd;

	class Nonlinearity {
		public:
			virtual ~Nonlinearity();

			virtual ArrayXXd operator()(const ArrayXXd& data) const = 0;
			virtual double operator()(double data) const = 0;
	};

	class InvertibleNonlinearity : virtual public Nonlinearity {
		public:
			virtual ArrayXXd inverse(const ArrayXXd& data) const = 0;
			virtual double inverse(double data) const = 0;
	};

	class DifferentiableNonlinearity : virtual public Nonlinearity {
		public:
			virtual ArrayXXd derivative(const ArrayXXd& data) const = 0;
	};

	class LogisticFunction : public InvertibleNonlinearity, public DifferentiableNonlinearity {
		public:
			LogisticFunction(double epsilon = 1e-50);

			virtual ArrayXXd operator()(const ArrayXXd& data) const;
			virtual double operator()(double data) const;

			virtual ArrayXXd derivative(const ArrayXXd& data) const;

			virtual ArrayXXd inverse(const ArrayXXd& data) const;
			virtual double inverse(double data) const;

		protected:
			double mEpsilon;
	};

	class ExponentialFunction : public InvertibleNonlinearity, public DifferentiableNonlinearity {
		public:
			ExponentialFunction();

			virtual ArrayXXd operator()(const ArrayXXd& data) const;
			virtual double operator()(double data) const;

			virtual ArrayXXd derivative(const ArrayXXd& data) const;

			virtual ArrayXXd inverse(const ArrayXXd& data) const;
			virtual double inverse(double data) const;
	};
}

#endif
