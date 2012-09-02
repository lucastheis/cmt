#ifndef CONDITIONALDISTRIBUTION_H
#define CONDITIONALDISTRIBUTION_H

#include "Eigen/Core"

using namespace Eigen;

class ConditionalDistribution {
	public:
		virtual int dimIn() const = 0;
		virtual int dimOut() const = 0;
		virtual Array<double, 1, Dynamic> logLikelihood(const MatrixXd& input, const MatrixXd& output) const = 0;
		virtual double evaluate(const MatrixXd& input, const MatrixXd& output) const;
};

#endif
