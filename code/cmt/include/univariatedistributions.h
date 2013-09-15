#ifndef CMT_UNIVARIATEDISTRIBUTIONS_H
#define CMT_UNIVARIATEDISTRIBUTIONS_H

#include "Eigen/Core"
#include "distribution.h"
#include "exception.h"

namespace CMT {
	using Eigen::Array;
	using Eigen::Dynamic;
	using Eigen::MatrixXd;

	class UnivariateDistribution : public Distribution {
		public:
			inline int dim() const;

			virtual double mean() const = 0;
			virtual void setMean(double mean) = 0;

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

			virtual double mean() const;
			virtual void setMean(double mean);

			virtual MatrixXd sample(int numSamples) const;
			virtual MatrixXd sample(
				const Array<double, 1, Dynamic>& means) const;

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

	class Poisson : public UnivariateDistribution {
		public:
			Poisson(double lambda = 1.);

			virtual double mean() const;
			virtual void setMean(double mean);

			virtual MatrixXd sample(int numSamples) const;
			virtual MatrixXd sample(
				const Array<double, 1, Dynamic>& means) const;

			virtual Array<double, 1, Dynamic> logLikelihood(
				const MatrixXd& data) const;
			virtual Array<double, 1, Dynamic> logLikelihood(
				const Array<double, 1, Dynamic>& data,
				const Array<double, 1, Dynamic>& means) const;

			virtual Array<double, 1, Dynamic> gradient(
				const Array<double, 1, Dynamic>& data,
				const Array<double, 1, Dynamic>& means) const;

		protected:
			double mLambda;
	};
}



inline int CMT::UnivariateDistribution::dim() const {
	return 1;
}



inline double CMT::Bernoulli::probability() const {
	return mProb;
}



inline void CMT::Bernoulli::setProbability(double prob) {
	if(prob < 0. || prob > 1.)
		throw Exception("Probability has to be between 0 and 1.");
	mProb = prob;
}

#endif
