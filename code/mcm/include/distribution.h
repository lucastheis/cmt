#ifndef DISTRIBUTION_H
#define DISTRIBUTION_H

#include "Eigen/Core"

using namespace Eigen;

class Distribution {
	public:
		virtual int dim() const = 0;
		virtual Array<double, 1, Dynamic> logLikelihood(const MatrixXd& data) const = 0;
		virtual double evaluate(const MatrixXd& data) const;
};

#endif
