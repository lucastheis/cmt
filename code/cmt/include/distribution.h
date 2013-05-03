#ifndef DISTRIBUTION_H
#define DISTRIBUTION_H

#include <vector>
using std::pair;

#include "Eigen/Core"
using Eigen::MatrixXd;
using Eigen::Array;
using Eigen::Dynamic;

class Distribution {
	public:
		virtual ~Distribution();

		virtual int dim() const = 0;

		virtual MatrixXd sample(int numSamples) const = 0;

		virtual Array<double, 1, Dynamic> logLikelihood(
			const MatrixXd& data) const = 0;
		virtual double evaluate(const MatrixXd& data) const;
};

#endif
