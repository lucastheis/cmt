#ifndef CMT_MCGSM_H
#define CMT_MCGSM_H

#include <vector>
#include <utility>
#include "Eigen/Core"
#include "trainable.h"
#include "exception.h"
#include "regularizer.h"

namespace CMT {
	using std::vector;
	using std::pair;

	using Eigen::Dynamic;
	using Eigen::Array;
	using Eigen::ArrayXXd;
	using Eigen::MatrixXd;

	class MCGSM : public Trainable {
		public:
			struct Parameters : public Trainable::Parameters {
				public:
					bool trainPriors;
					bool trainScales;
					bool trainWeights;
					bool trainFeatures;
					bool trainCholeskyFactors;
					bool trainPredictors;
					bool trainLinearFeatures;
					bool trainMeans;
					Regularizer regularizeFeatures;
					Regularizer regularizePredictors;
					Regularizer regularizeWeights;
					Regularizer regularizeLinearFeatures;
					Regularizer regularizeMeans;
					Regularizer regularizer;

					Parameters();
					Parameters(const Parameters& params);
					virtual Parameters& operator=(const Parameters& params);
			};

			using Trainable::logLikelihood;
			using Trainable::initialize;
			using Trainable::train;

			MCGSM(
				int dimIn,
				int dimOut = 1,
				int numComponents = 8,
				int numScales = 6,
				int numFeatures = -1);
			MCGSM(int dimIn, const MCGSM& mcgsm);
			MCGSM(int dimIn, int dimOut, const MCGSM& mcgsm);
			virtual ~MCGSM();

			inline int dimIn() const;
			inline int dimOut() const;
			inline int numComponents() const;
			inline int numScales() const;
			inline int numFeatures() const;

			inline ArrayXXd priors() const;
			inline void setPriors(const ArrayXXd& priors);

			inline ArrayXXd scales() const;
			inline void setScales(const ArrayXXd& scales);

			inline ArrayXXd weights() const;
			inline void setWeights(const ArrayXXd& weights);

			inline MatrixXd features() const;
			inline void setFeatures(const MatrixXd& features);

			inline vector<MatrixXd> choleskyFactors() const;
			inline void setCholeskyFactors(const vector<MatrixXd>& choleskyFactors);

			inline vector<MatrixXd> predictors() const;
			inline void setPredictors(const vector<MatrixXd>& predictors);

			inline MatrixXd linearFeatures() const;
			inline void setLinearFeatures(const MatrixXd& linearFeatures);

			inline MatrixXd means() const;
			inline void setMeans(const MatrixXd& means);

			virtual void initialize(const MatrixXd& input, const MatrixXd& output);

			virtual MatrixXd sample(const MatrixXd& input) const;
			virtual MatrixXd sample(
				const MatrixXd& input,
				const Array<int, 1, Dynamic>& labels) const;
			virtual MatrixXd reconstruct(const MatrixXd& input, const MatrixXd& output) const;
			virtual Array<int, 1, Dynamic> samplePrior(const MatrixXd& input) const;
			virtual Array<int, 1, Dynamic> samplePosterior(
				const MatrixXd& input,
				const MatrixXd& output) const;

			virtual ArrayXXd prior(const MatrixXd& input) const;
			virtual ArrayXXd posterior(const MatrixXd& input, const MatrixXd& output) const;

			virtual Array<double, 1, Dynamic> logLikelihood(
				const MatrixXd& input,
				const MatrixXd& output) const;
			virtual Array<double, 1, Dynamic> logLikelihood(
				const MatrixXd& input,
				const MatrixXd& output,
				const Array<int, 1, Dynamic>& labels) const;

			virtual pair<pair<ArrayXXd, ArrayXXd>, Array<double, 1, Dynamic> > computeDataGradient(
				const MatrixXd& input,
				const MatrixXd& output) const;

			virtual int numParameters(const Trainable::Parameters& params = Parameters()) const;
			virtual lbfgsfloatval_t* parameters(const Trainable::Parameters& params = Parameters()) const;
			virtual void setParameters(const lbfgsfloatval_t* x, const Trainable::Parameters& params = Parameters());
			virtual double parameterGradient(
				const MatrixXd& input,
				const MatrixXd& output,
				const lbfgsfloatval_t* x,
				lbfgsfloatval_t* g,
				const Trainable::Parameters& params = Parameters()) const;

		protected:
			// hyperparameters
			int mDimIn;
			int mDimOut;
			int mNumComponents;
			int mNumScales;
			int mNumFeatures;

			// parameters
			ArrayXXd mPriors;
			ArrayXXd mScales;
			ArrayXXd mWeights;
			MatrixXd mFeatures;
			vector<MatrixXd> mCholeskyFactors;
			vector<MatrixXd> mPredictors;
			MatrixXd mLinearFeatures;
			MatrixXd mMeans;

			virtual bool train(
				const MatrixXd& input,
				const MatrixXd& output,
				const MatrixXd* inputVal = 0,
				const MatrixXd* outputVal = 0,
				const Trainable::Parameters& params = Trainable::Parameters());
	};
}



inline int CMT::MCGSM::dimIn() const {
	return mDimIn;
}



inline int CMT::MCGSM::dimOut() const {
	return mDimOut;
}



inline int CMT::MCGSM::numComponents() const {
	return mNumComponents;
}



inline int CMT::MCGSM::numScales() const {
	return mNumScales;
}



inline int CMT::MCGSM::numFeatures() const {
	return mNumFeatures;
}



inline Eigen::ArrayXXd CMT::MCGSM::scales() const {
	return mScales;
}



inline void CMT::MCGSM::setScales(const ArrayXXd& scales) {
	if(scales.rows() != mNumComponents || scales.cols() != mNumScales)
		throw Exception("Wrong number of scales.");
	mScales = scales;
}



inline Eigen::ArrayXXd CMT::MCGSM::weights() const {
	return mWeights;
}



inline void CMT::MCGSM::setWeights(const ArrayXXd& weights) {
	if(weights.rows() != mNumComponents || weights.cols() != mNumFeatures)
		throw Exception("Wrong number of weights.");
	mWeights = weights;
}



inline Eigen::ArrayXXd CMT::MCGSM::priors() const {
	return mPriors;
}



inline void CMT::MCGSM::setPriors(const ArrayXXd& priors) {
	if(priors.rows() != mNumComponents || priors.cols() != mNumScales)
		throw Exception("Wrong number of prior weights.");
	mPriors = priors;
}



inline Eigen::MatrixXd CMT::MCGSM::features() const {
	return mFeatures;
}



inline void CMT::MCGSM::setFeatures(const MatrixXd& features) {
	if(features.rows() != mDimIn)
		throw Exception("Features have wrong dimensionality.");
	if(features.cols() != mNumFeatures)
		throw Exception("Wrong number of features.");
	mFeatures = features;
}



inline std::vector<Eigen::MatrixXd> CMT::MCGSM::choleskyFactors() const {
	return mCholeskyFactors;
}



inline void CMT::MCGSM::setCholeskyFactors(const vector<MatrixXd>& choleskyFactors) {
	if(choleskyFactors.size() != mNumComponents)
		throw Exception("Wrong number of Cholesky factors.");

	for(int i = 0; i < mNumComponents; ++i)
		if(choleskyFactors[i].rows() != mDimOut || choleskyFactors[i].cols() != mDimOut)
			throw Exception("Cholesky factor has wrong dimensionality.");

	mCholeskyFactors = choleskyFactors;

	#pragma omp parallel for
	for(int i = 0; i < mNumComponents; ++i) {
		double prec = mCholeskyFactors[i](0, 0);

		// normalize representation
		mCholeskyFactors[i] /= prec;
		mScales.row(i) += 2. * log(prec);
		mWeights.row(i) /= prec;
	}
}



inline std::vector<Eigen::MatrixXd> CMT::MCGSM::predictors() const {
	return mPredictors;
}



inline void CMT::MCGSM::setPredictors(const vector<MatrixXd>& predictors) {
	if(predictors.size() != mNumComponents)
		throw Exception("Wrong number of predictors.");

	for(int i = 0; i < predictors.size(); ++i)
		if(predictors[i].rows() != mDimOut || predictors[i].cols() != mDimIn)
			throw Exception("Predictor has wrong dimensionality.");

	mPredictors = predictors;
}



inline Eigen::MatrixXd CMT::MCGSM::linearFeatures() const {
	return mLinearFeatures;
}



inline void CMT::MCGSM::setLinearFeatures(const MatrixXd& linearFeatures) {
	if(linearFeatures.rows() != mNumComponents || linearFeatures.cols() != mDimIn)
		throw Exception("Linear features have wrong dimensionality.");
	mLinearFeatures = linearFeatures;
}



inline Eigen::MatrixXd CMT::MCGSM::means() const {
	return mMeans;
}



inline void CMT::MCGSM::setMeans(const MatrixXd& means) {
	if(means.cols() != mNumComponents || means.rows() != mDimOut)
		throw Exception("Means have wrong dimensionality.");
	mMeans = means;
}

#endif
