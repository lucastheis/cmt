#ifndef CMT_GLM_H
#define CMT_GLM_H

#include "Eigen/Core"
#include "trainable.h"
#include "distribution.h"
#include "nonlinearities.h"
#include "univariatedistributions.h"

namespace CMT {
	using Eigen::Array;
	using Eigen::Dynamic;
	using Eigen::VectorXd;

	/**
	 * A generic class for generalized linear models.
	 */
	class GLM : public Trainable {
		public:
			using Trainable::logLikelihood;

			GLM(
				int dimIn,
				Nonlinearity* nonlinearity,
				UnivariateDistribution* distribution);
			GLM(int dimIn);
			GLM(int dimIn, const GLM&);
			virtual ~GLM();

			inline int dimIn() const;
			inline int dimOut() const;

			inline Nonlinearity* nonlinearity() const;
			inline void setNonlinearity(Nonlinearity* nonlinearity);

			inline UnivariateDistribution* distribution() const;
			inline void setDistribution(UnivariateDistribution* distribution);

			inline VectorXd weights() const;
			inline void setWeights(const VectorXd& weights);

			inline double bias() const;
			inline void setBias(double bias);

			virtual Array<double, 1, Dynamic> logLikelihood(
				const MatrixXd& input,
				const MatrixXd& output) const;

			virtual MatrixXd sample(const MatrixXd& input) const;

			virtual pair<pair<ArrayXXd, ArrayXXd>, Array<double, 1, Dynamic> > computeDataGradient(
				const MatrixXd& input,
				const MatrixXd& output) const;

			virtual int numParameters(const Parameters& params = Parameters()) const;
			virtual lbfgsfloatval_t* parameters(
				const Parameters& params = Parameters()) const;
			virtual void setParameters(
				const lbfgsfloatval_t* x,
				const Parameters& params = Parameters());

			virtual double parameterGradient(
				const MatrixXd& input,
				const MatrixXd& output,
				const lbfgsfloatval_t* x,
				lbfgsfloatval_t* g,
				const Parameters& params) const;

		protected:
			int mDimIn;
			VectorXd mWeights;
			double mBias;
			Nonlinearity* mNonlinearity;
			UnivariateDistribution* mDistribution;
	};
}



inline int CMT::GLM::dimIn() const {
	return mDimIn;
}



inline int CMT::GLM::dimOut() const {
	return 1;
}



inline CMT::Nonlinearity* CMT::GLM::nonlinearity() const {
	return mNonlinearity;
}



inline void CMT::GLM::setNonlinearity(Nonlinearity* nonlinearity) {
	mNonlinearity = nonlinearity;
}



inline CMT::UnivariateDistribution* CMT::GLM::distribution() const {
	return mDistribution;
}



inline void CMT::GLM::setDistribution(UnivariateDistribution* distribution) {
	mDistribution = distribution;
}



inline Eigen::VectorXd CMT::GLM::weights() const {
	return mWeights;
}



inline void CMT::GLM::setWeights(const VectorXd& weights) {
	mWeights = weights;
}



inline double CMT::GLM::bias() const {
	return mBias;
}



inline void CMT::GLM::setBias(double bias) {
	mBias = bias;
}

#endif
