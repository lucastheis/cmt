#ifndef CMT_CONDITIONALDISTRIBUTION_H
#define CMT_CONDITIONALDISTRIBUTION_H

#include <map>
#include "Eigen/Core"
#include "preconditioner.h"

namespace CMT {
	using std::pair;

	using Eigen::MatrixXd;
	using Eigen::Array;
	using Eigen::Dynamic;
	using Eigen::ArrayXXd;

	class ConditionalDistribution {
		public:
			virtual ~ConditionalDistribution();

			virtual int dimIn() const = 0;
			virtual int dimOut() const = 0;
			virtual MatrixXd sample(const MatrixXd& input) const = 0;
			virtual MatrixXd predict(const MatrixXd& input) const;
			virtual Array<double, 1, Dynamic> logLikelihood(
				const pair<ArrayXXd, ArrayXXd>& data) const;
			virtual Array<double, 1, Dynamic> logLikelihood(
				const MatrixXd& input,
				const MatrixXd& output) const = 0;
			virtual double evaluate(const MatrixXd& input, const MatrixXd& output) const;
			virtual double evaluate(
					const MatrixXd& input,
					const MatrixXd& output,
					const Preconditioner& preconditioner) const;
			virtual double evaluate(const pair<ArrayXXd, ArrayXXd>& data) const;
			virtual double evaluate(
					const pair<ArrayXXd, ArrayXXd>& data,
					const Preconditioner& preconditioner) const;

			virtual pair<pair<ArrayXXd, ArrayXXd>, Array<double, 1, Dynamic> > computeDataGradient(
				const MatrixXd& input,
				const MatrixXd& output) const = 0;
	};
}

#endif
