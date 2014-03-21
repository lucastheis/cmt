#ifndef CMT_MCBM_H
#define CMT_MCBM_H

#include "Eigen/Core"
#include "exception.h"
#include "trainable.h"
#include "regularizer.h"

namespace CMT {
	using Eigen::VectorXd;
	using Eigen::MatrixXd;

	class MCBM : public Trainable {
		public:
			struct Parameters : public Trainable::Parameters {
				public:
					bool trainPriors;
					bool trainWeights;
					bool trainFeatures;
					bool trainPredictors;
					bool trainInputBias;
					bool trainOutputBias;
					Regularizer regularizeFeatures;
					Regularizer regularizePredictors;
					Regularizer regularizeWeights;

					Parameters();
					Parameters(const Parameters& params);
					virtual Parameters& operator=(const Parameters& params);
			};

			using Trainable::logLikelihood;
			using Trainable::train;

			MCBM(
				int dimIn, 
				int numComponents = 8,
				int numFeatures = -1);
			MCBM(int dimIn, const MCBM& mcbm);

			virtual ~MCBM();

			inline int dimIn() const;
			inline int dimOut() const;

			inline int numComponents() const;
			inline int numFeatures() const;

			inline VectorXd priors() const;
			inline void setPriors(const VectorXd& priors);

			inline MatrixXd weights() const;
			inline void setWeights(const MatrixXd& weights);

			inline MatrixXd features() const;
			inline void setFeatures(const MatrixXd& features);

			inline MatrixXd predictors() const;
			inline void setPredictors(const MatrixXd& predictors);

			inline MatrixXd inputBias() const;
			inline void setInputBias(const MatrixXd& inputBias);

			inline VectorXd outputBias() const;
			inline void setOutputBias(const VectorXd& outputBias);

			virtual MatrixXd sample(const MatrixXd& input) const;
			virtual Array<int, 1, Dynamic> samplePrior(const MatrixXd& input) const;
			virtual Array<int, 1, Dynamic> samplePosterior(
				const MatrixXd& input,
				const MatrixXd& output) const;

			virtual Array<double, 1, Dynamic> logLikelihood(
				const MatrixXd& input,
				const MatrixXd& output) const;

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
				const Trainable::Parameters& params = Parameters()) const;

			virtual pair<pair<ArrayXXd, ArrayXXd>, Array<double, 1, Dynamic> > computeDataGradient(
				const MatrixXd& input,
				const MatrixXd& output) const;

		protected:
			int mDimIn;
			int mNumComponents;
			int mNumFeatures;

			VectorXd mPriors;
			MatrixXd mWeights;
			MatrixXd mFeatures;
			MatrixXd mPredictors;
			MatrixXd mInputBias;
			VectorXd mOutputBias;

			virtual bool train(
				const MatrixXd& input,
				const MatrixXd& output,
				const MatrixXd* inputVal = 0,
				const MatrixXd* outputVal = 0,
				const Trainable::Parameters& params = Trainable::Parameters());
	};
}



inline int CMT::MCBM::dimIn() const {
	return mDimIn;
}



inline int CMT::MCBM::dimOut() const {
	return 1;
}



inline int CMT::MCBM::numComponents() const {
	return mNumComponents;
}



inline int CMT::MCBM::numFeatures() const {
	return mNumFeatures;
}



inline Eigen::MatrixXd CMT::MCBM::weights() const {
	return mWeights;
}



inline void CMT::MCBM::setWeights(const MatrixXd& weights) {
	if(weights.rows() != mNumComponents || weights.cols() != mNumFeatures)
		throw Exception("Wrong number of weights.");
	mWeights = weights;
}



inline Eigen::VectorXd CMT::MCBM::priors() const {
	return mPriors;
}



inline void CMT::MCBM::setPriors(const VectorXd& priors) {
	if(priors.size() != mNumComponents)
		throw Exception("Wrong number of prior weights.");
	mPriors = priors;
}



inline Eigen::MatrixXd CMT::MCBM::features() const {
	return mFeatures;
}



inline void CMT::MCBM::setFeatures(const MatrixXd& features) {
	if(features.rows() != mDimIn)
		throw Exception("Features have wrong dimensionality.");
	if(features.cols() != mNumFeatures)
		throw Exception("Wrong number of features.");
	mFeatures = features;
}



inline Eigen::MatrixXd CMT::MCBM::predictors() const {
	return mPredictors;
}



inline void CMT::MCBM::setPredictors(const MatrixXd& predictors) {
	if(predictors.cols() != mDimIn)
		throw Exception("Predictors have wrong dimensionality.");
	if(predictors.rows() != mNumComponents)
		throw Exception("Wrong number of predictors.");
	mPredictors = predictors;
}



inline Eigen::MatrixXd CMT::MCBM::inputBias() const {
	return mInputBias;
}



inline void CMT::MCBM::setInputBias(const MatrixXd& inputBias) {
	if(inputBias.rows() != mDimIn)
		throw Exception("Bias vectors have wrong dimensionality.");
	if(inputBias.cols() != mNumComponents)
		throw Exception("Wrong number of bias vectors.");
	mInputBias = inputBias;
}



inline Eigen::VectorXd CMT::MCBM::outputBias() const {
	return mOutputBias;
}



inline void CMT::MCBM::setOutputBias(const VectorXd& outputBias) {
	if(outputBias.size() != mNumComponents)
		throw Exception("Wrong number of biases.");
	mOutputBias = outputBias;
}

#endif
