#ifndef CMT_NONLINEARITIES_H
#define CMT_NONLINEARITIES_H

#include <vector>
#include "Eigen/Core"

namespace CMT {
	using Eigen::ArrayXXd;
	using Eigen::VectorXd;
	using std::vector;

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

	class TrainableNonlinearity : virtual public Nonlinearity {
		public:
			virtual VectorXd parameters() const = 0;
			virtual void setParameters(const VectorXd& parameters) = 0;

			virtual int numParameters(const ArrayXXd& data) const = 0;
			virtual ArrayXXd parameterGradient(const ArrayXXd& data) const = 0;
	};

	class LogisticFunction : public InvertibleNonlinearity, public DifferentiableNonlinearity {
		public:
			LogisticFunction(double epsilon = 1e-12);

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
			ExponentialFunction(double epsilon = 1e-12);

			virtual ArrayXXd operator()(const ArrayXXd& data) const;
			virtual double operator()(double data) const;

			virtual ArrayXXd derivative(const ArrayXXd& data) const;

			virtual ArrayXXd inverse(const ArrayXXd& data) const;
			virtual double inverse(double data) const;

		protected:
			double mEpsilon;
	};

	class HistogramNonlinearity : public Nonlinearity {
		public:
			HistogramNonlinearity(
				const ArrayXXd& inputs,
				const ArrayXXd& outputs,
				int numBins,
				double epsilon = 1e-12);
			HistogramNonlinearity(
				const ArrayXXd& inputs,
				const ArrayXXd& outputs,
				const vector<double>& binEdges,
				double epsilon = 1e-12);
			HistogramNonlinearity(
				const vector<double>& binEdges,
				double epsilon = 1e-12);

			virtual void initialize(
				const ArrayXXd& inputs,
				const ArrayXXd& outputs);
			virtual void initialize(
				const ArrayXXd& inputs,
				const ArrayXXd& outputs,
				int numBins);
			virtual void initialize(
				const ArrayXXd& inputs,
				const ArrayXXd& outputs,
				const vector<double>& binEdges);

			virtual ArrayXXd operator()(const ArrayXXd& inputs) const;
			virtual double operator()(double input) const;

		protected:
			double mEpsilon;
			vector<double> mBinEdges;
			vector<double> mHistogram;

			int bin(double input) const;
	};
}

#endif
