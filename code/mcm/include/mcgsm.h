#ifndef MCGSM_H
#define MCGSM_H

#include "Eigen/Core"
#include "conditionaldistribution.h"
#include "exception.h"
#include "lbfgs.h"
#include <iostream>
#include <cmath>
#include <vector>

using namespace Eigen;
using std::sqrt;
using std::vector;

class MCGSM : public ConditionalDistribution {
	public:
		struct Parameters {
			public:
				int verbosity;
				int maxIter;
				double tol;
				int numGrad;
				int batchSize;

				Parameters();
		};

		MCGSM(
			int dimIn, 
			int dimOut = 1,
			int numComponents = 8,
			int numScales = 6,
			int numFeatures = -1);
		virtual ~MCGSM();

		inline int dimIn() const;
		inline int dimOut() const;
		inline int numComponents() const;
		inline int numScales() const;
		inline int numFeatures() const;

		inline ArrayXXd priors() const;
		inline void setPriors(ArrayXXd priors);

		inline ArrayXXd scales() const;
		inline void setScales(ArrayXXd scales);

		inline ArrayXXd weights() const;
		inline void setWeights(ArrayXXd weights);

		inline MatrixXd features() const;
		inline void setFeatures(MatrixXd features);

		inline vector<MatrixXd> choleskyFactors() const;
		inline void setCholeskyFactors(vector<MatrixXd> choleskyFactors);

		inline vector<MatrixXd> predictors() const;
		inline void setPredictors(vector<MatrixXd> predictors);

		virtual void normalize();
		virtual bool train(
			const MatrixXd& input,
			const MatrixXd& output,
			Parameters params = Parameters());

		virtual double checkGradient(
			const MatrixXd& input,
			const MatrixXd& output,
			double epsilon = 1e-5,
			Parameters params = Parameters()) const;
		virtual double checkPerformance(
			const MatrixXd& input,
			const MatrixXd& output,
			int repetitions = 2,
			Parameters params = Parameters()) const;

		virtual MatrixXd sample(const MatrixXd& input) const;
		virtual Array<double, 1, Dynamic> samplePosterior(const MatrixXd& input, const MatrixXd& output) const;

		virtual ArrayXXd posterior(const MatrixXd& input, const MatrixXd& output) const;
		virtual Array<double, 1, Dynamic> logLikelihood(const MatrixXd& input, const MatrixXd& output) const;

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

		int numParameters() const;
		void copyParameters(lbfgsfloatval_t* x) const;
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



inline ArrayXXd MCGSM::scales() const {
	return mScales;
}



inline void MCGSM::setScales(ArrayXXd scales) {
	if(scales.rows() != mNumComponents || scales.cols() != mNumScales)
		throw Exception("Wrong number of scales.");
	mScales = scales;
}



inline ArrayXXd MCGSM::weights() const {
	return mWeights;
}



inline void MCGSM::setWeights(ArrayXXd weights) {
	if(weights.rows() != mNumComponents || weights.cols() != mNumFeatures)
		throw Exception("Wrong number of weights.");
	mWeights = weights;
}



inline ArrayXXd MCGSM::priors() const {
	return mPriors;
}



inline void MCGSM::setPriors(ArrayXXd priors) {
	if(priors.rows() != mNumComponents || priors.cols() != mNumScales)
		throw Exception("Wrong number of priors.");
	mPriors = priors;
}



inline MatrixXd MCGSM::features() const {
	return mFeatures;
}



inline void MCGSM::setFeatures(MatrixXd features) {
	if(features.rows() != mDimIn)
		throw Exception("Features have wrong dimensionality.");
	if(features.cols() != mNumFeatures)
		throw Exception("Wrong number of features.");
	mFeatures = features;
}



inline vector<MatrixXd> MCGSM::choleskyFactors() const {
	return mCholeskyFactors;
}



inline void MCGSM::setCholeskyFactors(vector<MatrixXd> choleskyFactors) {
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
		mScales.row(i) *= prec;
		mWeights.row(i) /= prec;
	}
}



inline vector<MatrixXd> MCGSM::predictors() const {
	return mPredictors;
}



inline void MCGSM::setPredictors(vector<MatrixXd> predictors) {
	if(predictors.size() != mNumComponents)
		throw Exception("Wrong number of predictors.");

	for(int i = 0; i < predictors.size(); ++i)
		if(predictors[i].rows() != mDimOut || predictors[i].cols() != mDimIn)
			throw Exception("Predictor has wrong dimensionality.");

	mPredictors = predictors;
}

#endif
