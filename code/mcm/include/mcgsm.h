#ifndef MCGSM_H
#define MCGSM_H

#include "Eigen/Core"
#include "conditionaldistribution.h"
#include "exception.h"
#include <iostream>
#include <cmath>
#include <vector>

using namespace Eigen;
using std::sqrt;
using std::vector;

class MCGSM : public ConditionalDistribution {
	public:
		MCGSM(
			int dimIn, 
			int dimOut = 1,
			int numComponents = 8,
			int numScales = 6,
			int numFeatures = -1);
		virtual ~MCGSM();

		inline int dimIn();
		inline int dimOut();
		inline int numComponents();
		inline int numScales();
		inline int numFeatures();

		inline ArrayXXd priors();
		inline void setPriors(ArrayXXd priors);

		inline ArrayXXd scales();
		inline void setScales(ArrayXXd scales);

		inline ArrayXXd weights();
		inline void setWeights(ArrayXXd weights);

		inline MatrixXd features();
		inline void setFeatures(MatrixXd features);

		inline vector<MatrixXd> choleskyFactors();
		inline void setCholeskyFactors(vector<MatrixXd> choleskyFactors);

		inline vector<MatrixXd> predictors();
		inline void setPredictors(vector<MatrixXd> predictors);

		virtual void normalize();
		virtual bool train(const MatrixXd& input, const MatrixXd& output, int maxIter = 100, double tol = 1e-5);
		virtual double checkGradient(const MatrixXd& input, const MatrixXd& output, double epsilon = 1e-5);

		virtual MatrixXd sample(const MatrixXd& input);
		virtual Array<double, 1, Dynamic> samplePosterior(const MatrixXd& input, const MatrixXd& output);

		virtual ArrayXXd posterior(const MatrixXd& input, const MatrixXd& output);
		virtual Array<double, 1, Dynamic> logLikelihood(const MatrixXd& input, const MatrixXd& output);

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



inline int MCGSM::dimIn() {
	return mDimIn;
}



inline int MCGSM::dimOut() {
	return mDimOut;
}



inline int MCGSM::numComponents() {
	return mNumComponents;
}



inline int MCGSM::numScales() {
	return mNumScales;
}



inline int MCGSM::numFeatures() {
	return mNumFeatures;
}



inline ArrayXXd MCGSM::scales() {
	return mScales;
}



inline void MCGSM::setScales(ArrayXXd scales) {
	if(scales.rows() != mNumComponents || scales.cols() != mNumScales)
		throw Exception("Wrong number of scales.");
	mScales = scales;
}



inline ArrayXXd MCGSM::weights() {
	return mWeights;
}



inline void MCGSM::setWeights(ArrayXXd weights) {
	if(weights.rows() != mNumComponents || weights.cols() != mNumFeatures)
		throw Exception("Wrong number of weights.");
	mWeights = weights;
}



inline ArrayXXd MCGSM::priors() {
	return mPriors;
}



inline void MCGSM::setPriors(ArrayXXd priors) {
	if(priors.rows() != mNumComponents || priors.cols() != mNumScales)
		throw Exception("Wrong number of priors.");
	mPriors = priors;
}



inline MatrixXd MCGSM::features() {
	return mFeatures;
}



inline void MCGSM::setFeatures(MatrixXd features) {
	if(features.rows() != mDimIn)
		throw Exception("Features have wrong dimensionality.");
	if(features.cols() != mNumFeatures)
		throw Exception("Wrong number of features.");
	mFeatures = features;
}



inline vector<MatrixXd> MCGSM::choleskyFactors() {
	return mCholeskyFactors;
}



inline void MCGSM::setCholeskyFactors(vector<MatrixXd> choleskyFactors) {
	if(choleskyFactors.size() != mNumComponents)
		throw Exception("Wrong number of Cholesky factors.");

	for(int i = 0; i < choleskyFactors.size(); ++i)
		if(choleskyFactors[i].rows() != mDimOut || choleskyFactors[i].cols() != mDimOut)
			throw Exception("Cholesky factor has wrong dimensionality.");

	mCholeskyFactors = choleskyFactors;
}



inline vector<MatrixXd> MCGSM::predictors() {
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
