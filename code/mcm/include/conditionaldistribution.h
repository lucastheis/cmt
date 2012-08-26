#ifndef DISTRIBUTION_H
#define DISTRIBUTION_H

#include "Eigen/Core"

using namespace Eigen;

class ConditionalDistribution {
	public:
		virtual int dimIn() = 0;
		virtual int dimOut() = 0;
		virtual Array<double, 1, Dynamic> logLikelihood(const MatrixXd& input, const MatrixXd& output) = 0;
		virtual double evaluate(const MatrixXd& input, const MatrixXd& output);
};

#endif
