#ifndef CMT_STM_H
#define CMT_STM_H

#include "Eigen/Core"
#include "exception.h"
#include "trainable.h"
#include "nonlinearities.h"
#include "univariatedistributions.h"
#include "regularizer.h"

namespace CMT {
	using Eigen::VectorXd;
	using Eigen::MatrixXd;

	class STM : public Trainable {
		public:
			struct Parameters : public Trainable::Parameters {
				public:
					bool trainSharpness;
					bool trainBiases;
					bool trainWeights;
					bool trainFeatures;
					bool trainPredictors;
					bool trainLinearPredictor;
					Regularizer regularizeBiases;
					Regularizer regularizeWeights;
					Regularizer regularizeFeatures;
					Regularizer regularizePredictors;
					Regularizer regularizeLinearPredictor;

					Parameters();
					Parameters(const Parameters& params);
					virtual Parameters& operator=(const Parameters& params);
			};

			using Trainable::logLikelihood;
			using Trainable::initialize;
			using Trainable::train;

			STM(
				int dimIn, 
				int numComponents = 3,
				int numFeatures = -1,
				Nonlinearity* nonlinearity = 0,
				UnivariateDistribution* distribution = 0);
			STM(
				int dimInNonlinear, 
				int dimInLinear, 
				int numComponents = 3,
				int numFeatures = -1,
				Nonlinearity* nonlinearity = 0,
				UnivariateDistribution* distribution = 0);
			STM(int dimIn, const STM& mcbm);

			inline int dimIn() const;
			inline int dimInNonlinear() const;
			inline int dimInLinear() const;
			inline int dimOut() const;

			inline int numComponents() const;
			inline int numFeatures() const;

			inline double sharpness() const;
			inline void setSharpness(double sharpness);

			inline VectorXd biases() const;
			inline void setBiases(const VectorXd& bias);

			inline MatrixXd weights() const;
			inline void setWeights(const MatrixXd& weights);

			inline MatrixXd features() const;
			inline void setFeatures(const MatrixXd& features);

			inline MatrixXd predictors() const;
			inline void setPredictors(const MatrixXd& predictors);

			inline VectorXd linearPredictor() const;
			inline void setLinearPredictor(const VectorXd& linearPredictor);

			inline Nonlinearity* nonlinearity() const;
			inline void setNonlinearity(Nonlinearity* nonlinearity);

			inline UnivariateDistribution* distribution() const;
			inline void setDistribution(UnivariateDistribution* distribution);

			virtual void initialize(const MatrixXd& input, const MatrixXd& output);

			virtual Array<double, 1, Dynamic> response(
				const MatrixXd& input) const;
			virtual Array<double, 1, Dynamic> response(
				const MatrixXd& inputNonlinear,
				const MatrixXd& inputLinear) const;
			virtual ArrayXXd nonlinearResponses(
				const MatrixXd& input) const;
			virtual ArrayXXd linearResponse(
				const MatrixXd& input) const;

			virtual MatrixXd sample(const MatrixXd& input) const;
			virtual MatrixXd sample(
				const MatrixXd& inputNonlinear,
				const MatrixXd& inputLinear) const;
			virtual MatrixXd predict(const MatrixXd& input) const;
			virtual MatrixXd predict(
				const MatrixXd& inputNonlinear,
				const MatrixXd& inputLinear) const;

			virtual Array<double, 1, Dynamic> logLikelihood(
				const MatrixXd& input,
				const MatrixXd& output) const;
			virtual Array<double, 1, Dynamic> logLikelihood(
				const MatrixXd& inputNonlinear,
				const MatrixXd& inputLinear,
				const MatrixXd& output) const;

			virtual bool train(
				const MatrixXd& inputNonlinear,
				const MatrixXd& inputLinear,
				const MatrixXd& output,
				const Parameters& params = Parameters());
			virtual bool train(
				const MatrixXd& inputNonlinear,
				const MatrixXd& inputLinear,
				const MatrixXd& output,
				const MatrixXd& inputNonlinearVal,
				const MatrixXd& inputLinearVal,
				const MatrixXd& outputVal,
				const Parameters& params = Parameters());

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
			static Nonlinearity* const defaultNonlinearity;
			static UnivariateDistribution* const defaultDistribution;

			int mDimInNonlinear;
			int mDimInLinear;
			int mNumComponents;
			int mNumFeatures;

			Nonlinearity* mNonlinearity;
			UnivariateDistribution* mDistribution;

			double mSharpness;
			VectorXd mBiases;
			MatrixXd mWeights;
			MatrixXd mFeatures;
			MatrixXd mPredictors;
			VectorXd mLinearPredictor;

			bool train(
				const MatrixXd& input,
				const MatrixXd& output,
				const MatrixXd* inputVal,
				const MatrixXd* outputVal,
				const Trainable::Parameters& params);
	};
}



inline int CMT::STM::dimIn() const {
	return mDimInNonlinear + mDimInLinear;
}



inline int CMT::STM::dimInNonlinear() const {
	return mDimInNonlinear;
}



inline int CMT::STM::dimInLinear() const {
	return mDimInLinear;
}



inline int CMT::STM::dimOut() const {
	return 1;
}



inline int CMT::STM::numComponents() const {
	return mNumComponents;
}



inline int CMT::STM::numFeatures() const {
	return mNumFeatures;
}



inline double CMT::STM::sharpness() const {
	return mSharpness;
}



inline void CMT::STM::setSharpness(double sharpness) {
	mSharpness = sharpness;
}



inline Eigen::MatrixXd CMT::STM::weights() const {
	return mWeights;
}



inline void CMT::STM::setWeights(const MatrixXd& weights) {
	if(weights.rows() != mNumComponents || weights.cols() != mNumFeatures)
		throw Exception("Wrong number of weights.");
	mWeights = weights;
}



inline Eigen::VectorXd CMT::STM::biases() const {
	return mBiases;
}



inline void CMT::STM::setBiases(const VectorXd& biases) {
	if(biases.size() != mNumComponents)
		throw Exception("Wrong number of biases.");
	mBiases = biases;
}



inline Eigen::MatrixXd CMT::STM::features() const {
	return mFeatures;
}



inline void CMT::STM::setFeatures(const MatrixXd& features) {
	if(features.rows() != dimInNonlinear())
		throw Exception("Features have wrong dimensionality.");
	if(features.cols() != mNumFeatures)
		throw Exception("Wrong number of features.");
	mFeatures = features;
}



inline Eigen::MatrixXd CMT::STM::predictors() const {
	return mPredictors;
}



inline void CMT::STM::setPredictors(const MatrixXd& predictors) {
	if(predictors.cols() != dimInNonlinear())
		throw Exception("Predictors have wrong dimensionality.");
	if(predictors.rows() != mNumComponents)
		throw Exception("Wrong number of predictors.");
	mPredictors = predictors;
}



inline Eigen::VectorXd CMT::STM::linearPredictor() const {
	return mLinearPredictor;
}



inline void CMT::STM::setLinearPredictor(const VectorXd& linearPredictor) {
	if(linearPredictor.size() != dimInLinear())
		throw Exception("Linear predictor has wrong dimensionality.");
	mLinearPredictor = linearPredictor;
}



inline CMT::Nonlinearity* CMT::STM::nonlinearity() const {
	return mNonlinearity;
}



inline void CMT::STM::setNonlinearity(Nonlinearity* nonlinearity) {
	mNonlinearity = nonlinearity;
}



inline CMT::UnivariateDistribution* CMT::STM::distribution() const {
	return mDistribution;
}



inline void CMT::STM::setDistribution(UnivariateDistribution* distribution) {
	mDistribution = distribution;
}

#endif
