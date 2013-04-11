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
		struct Parameters {
			public:
				virtual ~Parameters();
		};

		virtual int dimIn() const = 0;
		virtual int dimOut() const = 0;
		virtual MatrixXd sample(const MatrixXd& input) const = 0;
		virtual Array<double, 1, Dynamic> logLikelihood(
			const MatrixXd& input,
			const MatrixXd& output) const = 0;
		virtual double evaluate(const MatrixXd& input, const MatrixXd& output) const;

		virtual pair<pair<ArrayXXd, ArrayXXd>, Array<double, 1, Dynamic> > computeDataGradient(
			const MatrixXd& input,
			const MatrixXd& output) const = 0;
};

#endif
