#ifndef CMT_NONLINEARITIES_H
#define CMT_NONLINEARITIES_H

#include <vector>
#include "Eigen/Core"

namespace CMT {
	using Eigen::ArrayXXd;
	using Eigen::ArrayXd;
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
			virtual ArrayXd parameters() const = 0;
			virtual void setParameters(const ArrayXd& parameters) = 0;

			virtual int numParameters() const = 0;
			virtual ArrayXXd gradient(const ArrayXXd& data) const = 0;
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

	class HistogramNonlinearity : public TrainableNonlinearity {
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

			virtual ArrayXd parameters() const;
			virtual void setParameters(const ArrayXd& parameters);

			virtual int numParameters() const;
			virtual ArrayXXd gradient(const ArrayXXd& data) const;

		protected:
			double mEpsilon;
			vector<double> mBinEdges;
			vector<double> mHistogram;

			int bin(double input) const;
	};

	class BlobNonlinearity : public TrainableNonlinearity {
		public:
			BlobNonlinearity(
				int numComponents,
				double epsilon = 1e-12);

			virtual ArrayXXd operator()(const ArrayXXd& inputs) const;
			virtual double operator()(double input) const;

			virtual ArrayXd parameters() const;
			virtual void setParameters(const ArrayXd& parameters);

			virtual int numParameters() const;
			virtual ArrayXXd gradient(const ArrayXXd& data) const;

		protected:
			ArrayXd mMeans;
			ArrayXd mLogPrecisions;
			ArrayXd mLogWeights;
	};
}

#endif
