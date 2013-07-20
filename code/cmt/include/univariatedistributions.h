#ifndef CMT_UNIVARIATEDISTRIBUTIONS_H
#define CMT_UNIVARIATEDISTRIBUTIONS_H

#include "Eigen/Core"
#include "distribution.h"

namespace CMT {
	using Eigen::Array;
	using Eigen::Dynamic;
	using Eigen::MatrixXd;

	class UnivariateDistribution : public Distribution {
		public:
			inline int dim() const;

			/**
			 * Log-likelihood for different settings of the mean parameter.
			 *
			 * @param data data points for which to evaluate log-likelihood
			 * @param means parameters for which to evaluate log-likelihood
			 */
			virtual Array<double, 1, Dynamic> logLikelihood(
				const Array<double, 1, Dynamic>& data,
				const Array<double, 1, Dynamic>& means) const = 0;

			/**
			 * Generate sample using different parameter settings.
			 *
			 * @param data data points for which to evaluate gradient
			 * @param means parameters for which to evaluate gradient
			 */
			virtual MatrixXd sample(
				const Array<double, 1, Dynamic>& means) const = 0;

			/**
			 * Derivative of the *negative* log-likelihood with respect to the mean.
			 *
			 * @param data data points for which to evaluate gradient
			 * @param means parameters for which to evaluate gradient
			 */
			virtual Array<double, 1, Dynamic> gradient(
				const Array<double, 1, Dynamic>& data,
				const Array<double, 1, Dynamic>& means) const = 0;
	};

	class Bernoulli : public UnivariateDistribution {
		public:
			Bernoulli(double prob = 0.5);

			inline double probability() const;
			inline void setProbability(double prob);

			virtual MatrixXd sample(int numSamples) const;
			virtual MatrixXd sample(
				const Array<double, 1, Dynamic>& data) const;

			virtual Array<double, 1, Dynamic> logLikelihood(
				const MatrixXd& data) const;
			virtual Array<double, 1, Dynamic> logLikelihood(
				const Array<double, 1, Dynamic>& data,
				const Array<double, 1, Dynamic>& means) const;

			virtual Array<double, 1, Dynamic> gradient(
				const Array<double, 1, Dynamic>& data,
				const Array<double, 1, Dynamic>& means) const;

		protected:
			double mProb;
	};
}



inline int CMT::UnivariateDistribution::dim() const {
	return 1;
}



inline double CMT::Bernoulli::probability() const {
	return mProb;
}



inline void CMT::Bernoulli::setProbability(double prob) {
	mProb = prob;
}

#endif
