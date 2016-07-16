#ifndef MLR_H
#define MLR_H

#include "Eigen/Core"
#include "trainable.h"
#include "exception.h"
#include "regularizer.h"

namespace CMT {
	using Eigen::Array;
	using Eigen::Dynamic;
	using Eigen::VectorXd;
	using Eigen::MatrixXd;

	/**
	 * Multinomial logistic regression.
	 */
	class MLR : public Trainable {
		public:
			struct Parameters : public Trainable::Parameters {
				public:
					bool trainWeights;
					bool trainBiases;
					Regularizer regularizeWeights;
					Regularizer regularizeBiases;

					Parameters();
					Parameters(const Parameters& params);
					virtual Parameters& operator=(const Parameters& params);
			};

			using Trainable::logLikelihood;

			MLR(int dimIn, int dimOut);
			virtual ~MLR();

			inline int dimIn() const;
			inline int dimOut() const;

			inline MatrixXd weights() const;
			inline void setWeights(const MatrixXd& weights);

			inline VectorXd biases() const;
			inline void setBiases(const VectorXd& biases);

			virtual Array<double, 1, Dynamic> logLikelihood(
				const MatrixXd& input,
				const MatrixXd& output) const;

			virtual MatrixXd sample(const MatrixXd& input) const;
			virtual MatrixXd predict(const MatrixXd& input) const;

			virtual int numParameters(
				const Trainable::Parameters& params = Parameters()) const;
			virtual lbfgsfloatval_t* parameters(
				const Trainable::Parameters& params = Parameters()) const;
			virtual void setParameters(
				const lbfgsfloatval_t* x,
				const Trainable::Parameters& params = Parameters());
			virtual double parameterGradient(
				const MatrixXd& input,
				const MatrixXd& output,
				const lbfgsfloatval_t* x,
				lbfgsfloatval_t* g,
				const Trainable::Parameters& params) const;

			virtual pair<pair<ArrayXXd, ArrayXXd>, Array<double, 1, Dynamic> > computeDataGradient(
				const MatrixXd& input,
				const MatrixXd& output) const;

			virtual double evaluate(const MatrixXd& input, const MatrixXd& output) const;
			virtual double evaluate(
					const MatrixXd& input,
					const MatrixXd& output,
					const Preconditioner& preconditioner) const;
			virtual double evaluate(const pair<ArrayXXd, ArrayXXd>& data) const;
			virtual double evaluate(
					const pair<ArrayXXd, ArrayXXd>& data,
					const Preconditioner& preconditioner) const;

		private:
			int mDimIn;
			int mDimOut;

			MatrixXd mWeights;
			VectorXd mBiases;
	};
}



inline int CMT::MLR::dimIn() const {
	return mDimIn;
}



inline int CMT::MLR::dimOut() const {
	return mDimOut;
}



inline Eigen::MatrixXd CMT::MLR::weights() const {
	return mWeights;
}



inline void CMT::MLR::setWeights(const MatrixXd& weights) {
	if(weights.rows() != mDimOut || weights.cols() != mDimIn)
		throw Exception("Weight matrix has wrong dimensionality.");
	mWeights = weights;
}



inline Eigen::VectorXd CMT::MLR::biases() const {
	return mBiases;
}



inline void CMT::MLR::setBiases(const VectorXd& biases) {
	if(biases.size() != mDimOut)
		throw Exception("Wrong number of biases.");
	mBiases = biases;
}

#endif
