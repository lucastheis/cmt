#ifndef MCGSM_H
#define MCGSM_H

#include <cmath>
using std::sqrt;

#include <vector>
using std::vector;
using std::pair;

#include <utility>
using std::make_pair;

#include "Eigen/Core"
using Eigen::MatrixXd;
using Eigen::ArrayXXd;

#include "conditionaldistribution.h"
#include "exception.h"
#include "lbfgs.h"

class MCGSM : public ConditionalDistribution {
	public:
		struct Parameters : public ConditionalDistribution::Parameters {
			public:
				enum Regularizer { L1, L2 };

				bool trainPriors;
				bool trainScales;
				bool trainWeights;
				bool trainFeatures;
				bool trainCholeskyFactors;
				bool trainPredictors;
				double regularizeFeatures;
				double regularizePredictors;
				double regularizeWeights;
				Regularizer regularizer;

				Parameters();
				Parameters(const Parameters& params);
				virtual ~Parameters();
				virtual Parameters& operator=(const Parameters& params);
		};

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
		inline int numParameters(const Parameters& params = Parameters()) const;

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

		virtual void initialize(const MatrixXd& input, const MatrixXd& output);
		virtual bool train(
			const MatrixXd& input,
			const MatrixXd& output,
			const Parameters& params = Parameters());

		virtual double checkGradient(
			const MatrixXd& input,
			const MatrixXd& output,
			double epsilon = 1e-5,
			const Parameters& params = Parameters()) const;
		virtual double checkPerformance(
			const MatrixXd& input,
			const MatrixXd& output,
			int repetitions = 2,
			const Parameters& params = Parameters()) const;

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

		virtual pair<pair<ArrayXXd, ArrayXXd>, Array<double, 1, Dynamic> > computeDataGradient(
			const MatrixXd& input,
			const MatrixXd& output) const;

		lbfgsfloatval_t* parameters(const Parameters& params) const;
		void setParameters(const lbfgsfloatval_t* x, const Parameters& params);
		virtual double computeGradient(
			const MatrixXd& input,
			const MatrixXd& output,
			const lbfgsfloatval_t* x,
			lbfgsfloatval_t* g,
			const Parameters& params) const;

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
};



inline int MCGSM::dimIn() const {
	return mDimIn;
}



inline int MCGSM::dimOut() const {
	return mDimOut;
}



inline int MCGSM::numComponents() const {
	return mNumComponents;
}



inline int MCGSM::numScales() const {
	return mNumScales;
}



inline int MCGSM::numFeatures() const {
	return mNumFeatures;
}



inline int MCGSM::numParameters(const Parameters& params) const {
	int numParams = 0;
	if(params.trainPriors)
		numParams += mPriors.size();
	if(params.trainScales)
		numParams += mScales.size();
	if(params.trainWeights)
		numParams += mWeights.size();
	if(params.trainFeatures)
		numParams += mFeatures.size();
	if(params.trainCholeskyFactors)
		numParams += mNumComponents * mDimOut * (mDimOut + 1) / 2 - mNumComponents;
	if(params.trainPredictors)
		numParams += mNumComponents * mPredictors[0].size();
	return numParams;
}



inline ArrayXXd MCGSM::scales() const {
	return mScales;
}



inline void MCGSM::setScales(const ArrayXXd& scales) {
	if(scales.rows() != mNumComponents || scales.cols() != mNumScales)
		throw Exception("Wrong number of scales.");
	mScales = scales;
}



inline ArrayXXd MCGSM::weights() const {
	return mWeights;
}



inline void MCGSM::setWeights(const ArrayXXd& weights) {
	if(weights.rows() != mNumComponents || weights.cols() != mNumFeatures)
		throw Exception("Wrong number of weights.");
	mWeights = weights;
}



inline ArrayXXd MCGSM::priors() const {
	return mPriors;
}



inline void MCGSM::setPriors(const ArrayXXd& priors) {
	if(priors.rows() != mNumComponents || priors.cols() != mNumScales)
		throw Exception("Wrong number of prior weights.");
	mPriors = priors;
}



inline MatrixXd MCGSM::features() const {
	return mFeatures;
}



inline void MCGSM::setFeatures(const MatrixXd& features) {
	if(features.rows() != mDimIn)
		throw Exception("Features have wrong dimensionality.");
	if(features.cols() != mNumFeatures)
		throw Exception("Wrong number of features.");
	mFeatures = features;
}



inline vector<MatrixXd> MCGSM::choleskyFactors() const {
	return mCholeskyFactors;
}



inline void MCGSM::setCholeskyFactors(const vector<MatrixXd>& choleskyFactors) {
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



inline vector<MatrixXd> MCGSM::predictors() const {
	return mPredictors;
}



inline void MCGSM::setPredictors(const vector<MatrixXd>& predictors) {
	if(predictors.size() != mNumComponents)
		throw Exception("Wrong number of predictors.");

	for(int i = 0; i < predictors.size(); ++i)
		if(predictors[i].rows() != mDimOut || predictors[i].cols() != mDimIn)
			throw Exception("Predictor has wrong dimensionality.");

	mPredictors = predictors;
}

#endif
