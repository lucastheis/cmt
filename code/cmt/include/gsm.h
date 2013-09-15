#ifndef CMT_GSM_H
#define CMT_GSM_H

#include "Eigen/Core"
#include "Eigen/Cholesky"
#include "Eigen/LU"
#include "mixture.h"
#include "exception.h"
#include <cmath>

namespace CMT {
	using Eigen::VectorXd;
	using Eigen::MatrixXd;

	using std::pow;

	class GSM : public Mixture::Component {
		public:
			GSM(int dim = 1, int numScales = 6);

			virtual GSM* copy();
			virtual Mixture::Component& operator=(const Component& component);

			inline int dim() const;
			inline int numScales() const;

			inline VectorXd mean() const;
			inline void setMean(const VectorXd& mean);

			inline VectorXd priors() const;
			inline void setPriors(const VectorXd& priors);

			inline VectorXd scales() const;
			inline void setScales(const VectorXd& scales);

			inline MatrixXd cholesky() const;
			inline void setCholesky(const MatrixXd& cholesky);

			inline MatrixXd covariance() const;
			inline void setCovariance(const MatrixXd& covariance);

			virtual MatrixXd sample(int numSamples = 1) const;
			virtual Array<double, 1, Dynamic> logLikelihood(
				const MatrixXd& data) const;

			virtual void initialize(
				const MatrixXd& data,
				const Parameters& parameters = Parameters());
			virtual bool train(
				const MatrixXd& data,
				const Parameters& parameters = Parameters());
			virtual bool train(
				const MatrixXd& data,
				const Array<double, 1, Dynamic>& weights,
				const Parameters& parameters = Parameters());

		protected:
			int mDim;

			// mean of distribution
			VectorXd mMean;

			// prior probabilities of scales
			VectorXd mPriors;

			// precision scales
			VectorXd mScales;

			// Cholesky factor of precision matrix
			MatrixXd mCholesky;
	};
}



int CMT::GSM::dim() const {
	return mDim;
}



int CMT::GSM::numScales() const {
	return mScales.size();
}



Eigen::VectorXd CMT::GSM::mean() const {
	return mMean;
}



void CMT::GSM::setMean(const VectorXd& mean) {
	if(mean.size() != dim())
		throw Exception("Mean has wrong dimensionality.");
	mMean = mean;
}



Eigen::VectorXd CMT::GSM::priors() const {
	return mPriors;
}



void CMT::GSM::setPriors(const VectorXd& priors) {
	if(priors.size() != numScales())
		throw Exception("Wrong number of priors.");
	mPriors = priors.array().abs() / priors.array().abs().sum();
}



Eigen::VectorXd CMT::GSM::scales() const {
	return mScales;
}



void CMT::GSM::setScales(const VectorXd& scales) {
	if(scales.size() != numScales())
		throw Exception("Wrong number of scales.");
	mScales = scales;
}



Eigen::MatrixXd CMT::GSM::cholesky() const {
	return mCholesky;
}



void CMT::GSM::setCholesky(const MatrixXd& cholesky) {
	if(cholesky.rows() != dim() || cholesky.cols() != dim())
		throw Exception("Cholesky factor has wrong dimensionality.");
	mCholesky = cholesky;
}



Eigen::MatrixXd CMT::GSM::covariance() const {
	MatrixXd choleskyInv = mCholesky.triangularView<Eigen::Lower>().solve(
		MatrixXd::Identity(dim(), dim()));
	return choleskyInv.transpose() * choleskyInv;
}



void CMT::GSM::setCovariance(const MatrixXd& covariance) {
	if(covariance.rows() != dim() || covariance.cols() != dim())
		throw Exception("Covariance matrix has wrong dimensionality.");

	// compute Cholesky factor of precision matrix
	mCholesky = covariance.inverse().llt().matrixL();
}

#endif
