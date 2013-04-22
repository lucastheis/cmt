#ifndef CONDITIONALDISTRIBUTION_H
#define CONDITIONALDISTRIBUTION_H

#include <vector>
using std::pair;

#include "Eigen/Core"
using Eigen::MatrixXd;
using Eigen::Array;
using Eigen::Dynamic;
using Eigen::ArrayXXd;

class ConditionalDistribution {
	public:
		class Callback {
			public:
				virtual ~Callback();
				virtual Callback* copy() = 0;
				virtual bool operator()(int iter, const ConditionalDistribution& cd) = 0;
		};

		struct Parameters {
			public:
				int verbosity;
				int maxIter;
				double threshold;
				int numGrad;
				int batchSize;
				Callback* callback;
				int cbIter;
				int valIter;

				ArrayXXd* valInput;
				ArrayXXd* valOutput;

				Parameters();
				Parameters(const Parameters& params);
				virtual ~Parameters();
				virtual Parameters& operator=(const Parameters& params);
		};

		virtual ~ConditionalDistribution();

		virtual int dimIn() const = 0;
		virtual int dimOut() const = 0;
		virtual MatrixXd sample(const MatrixXd& input) const = 0;
		virtual Array<double, 1, Dynamic> logLikelihood(
			const MatrixXd& input,
			const MatrixXd& output) const = 0;
		virtual double evaluate(const MatrixXd& input, const MatrixXd& output) const;

		virtual void initialize(const MatrixXd& input, const MatrixXd& output) const;
		virtual bool train(const MatrixXd& input, const MatrixXd& output) const;

		virtual pair<pair<ArrayXXd, ArrayXXd>, Array<double, 1, Dynamic> > computeDataGradient(
			const MatrixXd& input,
			const MatrixXd& output) const = 0;
};

#endif
