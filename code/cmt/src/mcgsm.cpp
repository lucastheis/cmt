#include "mcgsm.h"
#include "utils.h"
#include "mogsm.h"

#include <utility>
using std::pair;
using std::make_pair;

#include "Eigen/Core"
using Eigen::Dynamic;
using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::Array;
using Eigen::ArrayXXd;
using Eigen::ArrayXd;
using Eigen::Map;

#include <cmath>
using std::max;
using std::min;

#include "Eigen/Eigenvalues"
using Eigen::SelfAdjointEigenSolver;

#include <iostream>
using std::cout;
using std::endl;

CMT::MCGSM::Parameters::Parameters() :
	Trainable::Parameters::Parameters(),
	trainPriors(true),
	trainScales(true),
	trainWeights(true),
	trainFeatures(true),
	trainCholeskyFactors(true),
	trainPredictors(true),
	trainLinearFeatures(false),
	trainMeans(false),
	regularizeFeatures(0.),
	regularizePredictors(0.),
	regularizeWeights(0.),
	regularizeLinearFeatures(0.),
	regularizeMeans(0.)
{
}



CMT::MCGSM::Parameters::Parameters(const Parameters& params) :
	Trainable::Parameters::Parameters(params),
	trainPriors(params.trainPriors),
	trainScales(params.trainScales),
	trainWeights(params.trainWeights),
	trainFeatures(params.trainFeatures),
	trainCholeskyFactors(params.trainCholeskyFactors),
	trainPredictors(params.trainPredictors),
	trainLinearFeatures(params.trainLinearFeatures),
	trainMeans(params.trainMeans),
	regularizeFeatures(params.regularizeFeatures),
	regularizePredictors(params.regularizePredictors),
	regularizeWeights(params.regularizeWeights),
	regularizeLinearFeatures(params.regularizeLinearFeatures),
	regularizeMeans(params.regularizeMeans)
{
}



CMT::MCGSM::Parameters& CMT::MCGSM::Parameters::operator=(const Parameters& params) {
	Trainable::Parameters::operator=(params);

	trainPriors = params.trainPriors;
	trainScales = params.trainScales;
	trainWeights = params.trainWeights;
	trainFeatures = params.trainFeatures;
	trainCholeskyFactors = params.trainCholeskyFactors;
	trainPredictors = params.trainPredictors;
	trainLinearFeatures = params.trainLinearFeatures;
	trainMeans = params.trainMeans;
	regularizeFeatures = params.regularizeFeatures;
	regularizePredictors = params.regularizePredictors;
	regularizeWeights = params.regularizeWeights;
	regularizeLinearFeatures = params.regularizeLinearFeatures;
	regularizeWeights = params.regularizeWeights;

	return *this;
}



CMT::MCGSM::MCGSM(
	int dimIn,
	int dimOut,
	int numComponents,
	int numScales,
	int numFeatures) :
	mDimIn(dimIn),
	mDimOut(dimOut),
	mNumComponents(numComponents),
	mNumScales(numScales),
	mNumFeatures(numFeatures < 0 ? dimIn : numFeatures)
{
	// check hyperparameters
	if(mDimIn < 0)
		throw Exception("The number of input dimensions has to greater or equal zero.");
	if(mDimOut < 1)
		throw Exception("The number of output dimensions has to be positive.");
	if(mNumScales < 1)
		throw Exception("The number of scales has to be positive.");
	if(mNumComponents < 1)
		throw Exception("The number of components has to be positive.");

	if(mDimIn < 1)
		mNumFeatures = 0;

	// initialize parameters
	mPriors = ArrayXXd::Zero(mNumComponents, mNumScales);
	mScales = ArrayXXd::Random(mNumComponents, mNumScales);
	mWeights = ArrayXXd::Random(mNumComponents, mNumFeatures).abs() / 100. + 0.01;
	mFeatures = sampleNormal(mDimIn, mNumFeatures) / 100.;
	mLinearFeatures = MatrixXd::Zero(mNumComponents, mDimIn);
	mMeans = MatrixXd::Zero(mDimOut, mNumComponents);

	for(int i = 0; i < mNumComponents; ++i) {
		mCholeskyFactors.push_back(MatrixXd::Identity(mDimOut, mDimOut));
		mPredictors.push_back(sampleNormal(mDimOut, mDimIn) / 10.);
	}
}



CMT::MCGSM::MCGSM(int dimIn, int dimOut, const MCGSM& mcgsm) :
	mDimIn(dimIn),
	mDimOut(dimOut),
	mNumComponents(mcgsm.numComponents()),
	mNumScales(mcgsm.numScales()),
	mNumFeatures(mcgsm.numFeatures())
{
	// check hyperparameters
	if(mDimIn < 0)
		throw Exception("The number of input dimensions has to greater or equal zero.");
	if(mDimOut < 1)
		throw Exception("The number of output dimensions has to be positive.");

	if(mDimIn < 1)
		mNumFeatures = 0;

	// initialize parameters
	mPriors = ArrayXXd::Zero(mNumComponents, mNumScales);
	mScales = ArrayXXd::Random(mNumComponents, mNumScales);
	mWeights = ArrayXXd::Random(mNumComponents, mNumFeatures).abs() / 100. + 0.01;
	mFeatures = sampleNormal(mDimIn, mNumFeatures) / 100.;
	mLinearFeatures = MatrixXd::Zero(mNumComponents, mDimIn);
	mMeans = MatrixXd::Zero(mDimOut, mNumComponents);

	for(int i = 0; i < mNumComponents; ++i) {
		mCholeskyFactors.push_back(MatrixXd::Identity(mDimOut, mDimOut));
		mPredictors.push_back(sampleNormal(mDimOut, mDimIn) / 10.);
	}
}



CMT::MCGSM::MCGSM(int dimIn, const MCGSM& mcgsm) :
	mDimIn(dimIn),
	mDimOut(mcgsm.dimOut()),
	mNumComponents(mcgsm.numComponents()),
	mNumScales(mcgsm.numScales()),
	mNumFeatures(mcgsm.numFeatures())
{
	// initialize parameters
	mPriors = ArrayXXd::Zero(mNumComponents, mNumScales);
	mScales = ArrayXXd::Random(mNumComponents, mNumScales);
	mWeights = ArrayXXd::Random(mNumComponents, mNumFeatures).abs() / 100. + 0.01;
	mFeatures = sampleNormal(mDimIn, mNumFeatures) / 100.;
	mLinearFeatures = MatrixXd::Zero(mNumComponents, mDimIn);
	mMeans = MatrixXd::Zero(mDimOut, mNumComponents);

	for(int i = 0; i < mNumComponents; ++i) {
		mCholeskyFactors.push_back(MatrixXd::Identity(mDimOut, mDimOut));
		mPredictors.push_back(sampleNormal(mDimOut, mDimIn) / 10.);
	}
}



CMT::MCGSM::~MCGSM() {
}



void CMT::MCGSM::initialize(const MatrixXd& input, const MatrixXd& output) {
	if(input.rows() != mDimIn || output.rows() != mDimOut)
		throw Exception("Data has wrong dimensionality.");
	if(mDimIn && input.cols() != output.cols())
		throw Exception("The number of inputs and outputs should be the same.");

	if(!mDimIn) {
		// MCGSM reduces to MoGSM for zero-dimensional inputs
		MoGSM mogsm(mDimOut, mNumComponents, mNumScales);

		// initialize model
		ArrayXd priors = mPriors.exp().rowwise().sum();
		mogsm.setPriors(priors / priors.sum());

		for(int k = 0; k < mNumComponents; ++k) {
			ArrayXd priors = mPriors.row(k).exp().transpose();
			GSM* gsm = dynamic_cast<GSM*>(mogsm[k]);
			gsm->setMean(VectorXd::Zero(mDimOut));
			gsm->setPriors(priors / priors.sum());
			gsm->setScales(mScales.row(k).exp().transpose());
			gsm->setCholesky(mCholeskyFactors[k]);
		}

		// hyperparameters
		MoGSM::Parameters mogsmParams;
		MoGSM::Component::Parameters gsmParams;
		gsmParams.trainMean = false;

		// initialize mixture of GSMs
		mogsm.initialize(output, mogsmParams, gsmParams);

		// copy parameters back
		mPriors.colwise() = mogsm.priors().array().log();
		mMeans.setZero();

		vector<MatrixXd> choleskyFactors;

		for(int k = 0; k < mNumComponents; ++k) {
			GSM* gsm = dynamic_cast<GSM*>(mogsm[k]);

			mPriors.row(k) += gsm->priors().array().log().transpose();
			mScales.row(k) = gsm->scales().array().log().transpose();
			choleskyFactors.push_back(gsm->cholesky());
		}

		setCholeskyFactors(choleskyFactors);
	} else {
		MatrixXd covXX = covariance(input);
		MatrixXd covXY = covariance(input, output);

		MatrixXd whitening = SelfAdjointEigenSolver<MatrixXd>(covXX).operatorInverseSqrt();

		mScales = sampleNormal(mNumComponents, mNumScales) / 20.;
		mWeights = ArrayXXd::Random(mNumComponents, mNumFeatures).abs() / 100. + 0.01;
		mFeatures = whitening.transpose() * sampleNormal(mDimIn, mNumFeatures).matrix() / 100.;

		mLinearFeatures.setZero();
		mMeans.setZero();

		// optimal linear predictor and precision
		MatrixXd predictor = covXY.transpose() * covXX.inverse();
		MatrixXd choleskyFactor = covariance(output - predictor * input).inverse().llt().matrixL();
		vector<MatrixXd> choleskyFactors;

		for(int i = 0; i < mNumComponents; ++i) {
			mPredictors[i] = predictor + sampleNormal(mDimOut, mDimIn).matrix() / 10.;
			choleskyFactors.push_back(choleskyFactor);
		}

		setCholeskyFactors(choleskyFactors);
	}
}



MatrixXd CMT::MCGSM::sample(const MatrixXd& input) const {
	// initialize samples with Gaussian noise
	MatrixXd output = sampleNormal(mDimOut, input.cols());

	ArrayXXd weightsOutput;
	ArrayXXd scalesExp = mScales.exp();

	if(mDimIn) {
		ArrayXXd featuresOutput = mFeatures.transpose() * input;
		weightsOutput = mWeights.square().matrix() * featuresOutput.square().matrix()
			- 2. * mLinearFeatures * input;
	}

	#pragma omp parallel for
	for(int k = 0; k < input.cols(); ++k) {
		// compute joint distribution over components and scales
		ArrayXXd pmf;

		if(mDimIn)
			pmf = (mPriors - scalesExp.colwise() * weightsOutput.col(k) / 2.).exp();
		else
			pmf = mPriors.exp();

		pmf /= pmf.sum();

		// sample component and scale
		double urand = static_cast<double>(rand()) / (RAND_MAX + 1.);
		double cdf;
		int l = 0;

		for(cdf = pmf(0, 0); cdf < urand; cdf += pmf(l / mNumScales, l % mNumScales))
			++l;

		// component and scale index
		int i = l / mNumScales;
		int j = l % mNumScales;

		// apply precision matrix
		mCholeskyFactors[i].transpose().triangularView<Eigen::Upper>().solveInPlace(output.col(k));

		// apply scale
		output.col(k) /= sqrt(scalesExp(i, j));

		// add mean
		output.col(k) += mMeans.col(i);
		if(mDimIn)
			output.col(k) += mPredictors[i] * input.col(k);
	}

	return output;
}



MatrixXd CMT::MCGSM::sample(
	const MatrixXd& input,
	const Array<int, 1, Dynamic>& labels) const 
{
	if(input.rows() != mDimIn)
		throw Exception("Data has wrong dimensionality.");
	if(input.cols() != labels.cols())
		throw Exception("The number of inputs and labels should be the same.");

	MatrixXd output = sampleNormal(mDimOut, input.cols());

	MatrixXd scalesExp = mScales.exp();
	ArrayXXd featuresOutput;
	MatrixXd weightsSqr;

	if(mDimIn) {
		featuresOutput = mFeatures.transpose() * input;
		weightsSqr = mWeights.square();
	}

	#pragma omp parallel for
	for(int i = 0; i < input.cols(); ++i) {
		int k = labels[i];

		// compute distribution over scales
		ArrayXd pmf;

		if(mDimIn)
			pmf = mPriors.row(k).matrix() -
				scalesExp.row(k) * (
					weightsSqr.row(k) * featuresOutput.col(i).square().matrix() / 2. -
					mLinearFeatures.row(k) * input.col(i))[0];
		else
			pmf = mPriors.row(k);

		pmf = (pmf - logSumExp(pmf)[0]).exp();

		// sample scale
		double urand = static_cast<double>(rand()) / (RAND_MAX + 1.);
		double cdf;
		int j = 0;

		for(cdf = pmf(0); cdf < urand; cdf += pmf(j))
			++j;

		// apply precision matrix
		mCholeskyFactors[k].transpose().triangularView<Eigen::Upper>().solveInPlace(output.col(i));

		// apply scale
		output.col(i) /= sqrt(scalesExp(k, j));

		// add predicted mean
		output.col(i) += mMeans.col(k);
		if(mDimIn)
			output.col(i) += mPredictors[k] * input.col(i);
	}

	return output;
}



MatrixXd CMT::MCGSM::reconstruct(const MatrixXd& input, const MatrixXd& output) const {
	// reconstruct output from sampled labels
	return sample(input, samplePosterior(input, output));
}



Array<int, 1, Dynamic> CMT::MCGSM::samplePrior(const MatrixXd& input) const {
	if(input.rows() != mDimIn)
		throw Exception("Data has wrong dimensionality.");

	Array<int, 1, Dynamic> labels(input.cols());
	ArrayXXd pmf = prior(input);

	#pragma omp parallel for
	for(int j = 0; j < input.cols(); ++j) {
		int i = 0;
		double urand = static_cast<double>(rand()) / (RAND_MAX + 1.);
		double cdf;

		// compute index
		for(cdf = pmf(0, j); cdf < urand; cdf += pmf(i, j))
			++i;

		labels[j] = i;
	}

	return labels;
}



Array<int, 1, Dynamic> CMT::MCGSM::samplePosterior(const MatrixXd& input, const MatrixXd& output) const {
	if(input.rows() != mDimIn || output.rows() != mDimOut)
		throw Exception("Data has wrong dimensionality.");
	if(input.cols() != output.cols())
		throw Exception("The number of inputs and outputs should be the same.");

	Array<int, 1, Dynamic> labels(input.cols());
	ArrayXXd pmf = posterior(input, output);

	#pragma omp parallel for
	for(int j = 0; j < input.cols(); ++j) {
		int i = 0;
		double urand = static_cast<double>(rand()) / (RAND_MAX + 1.);
		double cdf;

		// compute index
		for(cdf = pmf(0, j); cdf < urand; cdf += pmf(i, j))
			++i;

		labels[j] = i;
	}

	return labels;
}



ArrayXXd CMT::MCGSM::prior(const MatrixXd& input) const {
	if(input.rows() != mDimIn)
		throw Exception("Data has wrong dimensionality.");

	ArrayXXd prior(mNumComponents, input.cols());

	MatrixXd weightsOutput;
	MatrixXd scalesExp = mScales.exp().transpose();

	if(mDimIn) {
		ArrayXXd featuresOutput = mFeatures.transpose() * input;
		weightsOutput = mWeights.square().matrix() * featuresOutput.square().matrix()
			- 2. * mLinearFeatures * input;
	}

	#pragma omp parallel for
	for(int i = 0; i < mNumComponents; ++i) {
		// compute unnormalized posterior
		ArrayXXd negEnergy(mNumScales, input.cols());
		
		if(mDimIn) {
			negEnergy = -scalesExp.col(i) / 2. * weightsOutput.row(i);
			negEnergy.colwise() += mPriors.row(i).transpose();
		} else {
			negEnergy.colwise() = mPriors.row(i).transpose();
		}

		// marginalize out scales
		prior.row(i) = logSumExp(negEnergy);
	}

	// return normalized prior
	return (prior.rowwise() - logSumExp(prior)).exp();
}



ArrayXXd CMT::MCGSM::posterior(const MatrixXd& input, const MatrixXd& output) const {
	if(input.rows() != mDimIn || output.rows() != mDimOut)
		throw Exception("Data has wrong dimensionality.");
	if(input.cols() != output.cols())
		throw Exception("The number of inputs and outputs should be the same.");

	ArrayXXd posterior(mNumComponents, input.cols());

	MatrixXd weightsOutput;
	MatrixXd scalesExp = mScales.array().exp().transpose();

	if(mDimIn) {
		ArrayXXd featuresOutput = mFeatures.transpose() * input;
		weightsOutput = mWeights.square().matrix() * featuresOutput.square().matrix()
			- 2. * mLinearFeatures * input;
	}

	#pragma omp parallel for
	for(int i = 0; i < mNumComponents; ++i) {
		Matrix<double, 1, Dynamic> errorSqr;
		ArrayXXd negEnergy;

		// compute unnormalized posterior
		if(mDimIn) {
			errorSqr = (mCholeskyFactors[i].transpose() *
				((output - mPredictors[i] * input).colwise() - mMeans.col(i))).colwise().squaredNorm();
			negEnergy = -scalesExp.col(i) / 2. * (weightsOutput.row(i) + errorSqr);
		} else {
			errorSqr = (mCholeskyFactors[i].transpose() * (output.colwise() - mMeans.col(i))).colwise().squaredNorm();
			negEnergy = -scalesExp.col(i) / 2. * errorSqr;
		}

		// normalization constants of experts
		double logDet = mCholeskyFactors[i].diagonal().array().abs().log().sum();
		ArrayXd logPartf = mDimOut * mScales.row(i).array() / 2. + logDet;
		negEnergy.colwise() += mPriors.row(i).transpose() + logPartf;

		// marginalize out scales
		posterior.row(i) = logSumExp(negEnergy);
	}

	// return normalized posterior
	return (posterior.rowwise() - logSumExp(posterior)).exp();
}



Array<double, 1, Dynamic> CMT::MCGSM::logLikelihood(
	const MatrixXd& input,
	const MatrixXd& output) const
{
	if(input.rows() != mDimIn || output.rows() != mDimOut)
		throw Exception("Data has wrong dimensionality.");
	if(mDimIn && input.cols() != output.cols())
		throw Exception("The number of inputs and outputs should be the same.");

	ArrayXXd logLikelihood(mNumComponents, input.cols());
	ArrayXXd normConsts(mNumComponents, input.cols());

	MatrixXd weightsOutput;
	MatrixXd scalesExp = mScales.array().exp().transpose();

	if(mDimIn) {
		ArrayXXd featuresOutput = mFeatures.transpose() * input;
		weightsOutput = mWeights.square().matrix() * featuresOutput.square().matrix()
			- 2. * mLinearFeatures * input;
	}

	#pragma omp parallel for
	for(int i = 0; i < mNumComponents; ++i) {
		ArrayXXd negEnergy(mNumScales, output.cols());
		MatrixXd outputWhitened;

		// compute gate energy
		if(mDimIn) {
			negEnergy = -scalesExp.col(i) / 2. * weightsOutput.row(i);
			negEnergy.colwise() += mPriors.row(i).transpose();
			outputWhitened = mCholeskyFactors[i].transpose() * ((output - mPredictors[i] * input).colwise() - mMeans.col(i));
		} else {
			negEnergy.colwise() = mPriors.row(i).transpose();
			outputWhitened = mCholeskyFactors[i].transpose() * (output.colwise() - mMeans.col(i));
		}

		// normalization constants of gates
		normConsts.row(i) = logSumExp(negEnergy);

		// compute expert energy
		negEnergy -= (scalesExp.col(i) / 2. * outputWhitened.colwise().squaredNorm()).array();

		// normalization constants of experts
		double logDet = mCholeskyFactors[i].diagonal().array().abs().log().sum();
		ArrayXd logPartf = mDimOut / 2. * mScales.row(i).array() +
			logDet - mDimOut / 2. * log(2. * PI);
		negEnergy.colwise() += logPartf;

		// marginalize out scales
		logLikelihood.row(i) = logSumExp(negEnergy);
	}

	// marginalize out components
	return logSumExp(logLikelihood) - logSumExp(normConsts);
}



Array<double, 1, Dynamic> CMT::MCGSM::logLikelihood(
	const MatrixXd& input,
	const MatrixXd& output,
	const Array<int, 1, Dynamic>& labels) const
{
	if(input.rows() != mDimIn || output.rows() != mDimOut)
		throw Exception("Data has wrong dimensionality.");
	if(mDimIn && input.cols() != output.cols())
		throw Exception("The number of inputs and outputs should be the same.");
	if(output.cols() != labels.cols())
		throw Exception("The number of outputs and labels should be the same.");

	ArrayXXd logLikelihood(mNumScales, output.cols());
	ArrayXXd scalesExp = mScales.array().exp();
	ArrayXd logPartf(mNumComponents);
	ArrayXXd featuresOutput;
	MatrixXd weightsSqr;

	if(mDimIn) {
		featuresOutput = mFeatures.transpose() * input;
		weightsSqr = mWeights.square();
	}

	#pragma omp parallel for
	for(int k = 0; k < mNumComponents; ++k)
		logPartf[k] = mCholeskyFactors[k].diagonal().array().abs().log().sum()
			- mDimOut / 2. * log(2. * PI);

	#pragma omp parallel for
	for(int i = 0; i < output.cols(); ++i) {
		int k = labels[i];

		// compute distribution over scales
		ArrayXd logPrior;
		VectorXd outputWhitened;

		if(mDimIn) {
			logPrior = mPriors.row(k) - scalesExp.row(k) * (
				weightsSqr.row(k) * featuresOutput.col(i).square().matrix() / 2. -
				mLinearFeatures.row(k) * input.col(i))[0];
			outputWhitened =
				mCholeskyFactors[k] * (output.col(i) - mPredictors[k] * input.col(i) - mMeans.col(k));
		} else {
			logPrior = mPriors.row(k);
			outputWhitened =
				mCholeskyFactors[k] * (output.col(i) - mMeans.col(k));
		}

		// normalize
		logPrior = logPrior - logSumExp(logPrior)[0];

		logLikelihood.col(i) = logPrior
			+ mDimOut / 2. * mScales.row(k).transpose()
			- outputWhitened.squaredNorm() / 2. * scalesExp.row(k).transpose()
			+ logPartf[k];
	}

	// marginalize out scales
	return logSumExp(logLikelihood);
}



int CMT::MCGSM::numParameters(const Trainable::Parameters& params_) const {
	const Parameters& params = dynamic_cast<const Parameters&>(params_);

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
	if(params.trainLinearFeatures)
		numParams += mLinearFeatures.size();
	if(params.trainMeans)
		numParams += mMeans.size();

	return numParams;
}



lbfgsfloatval_t* CMT::MCGSM::parameters(const Trainable::Parameters& params_) const {
	const Parameters& params = dynamic_cast<const Parameters&>(params_);

	lbfgsfloatval_t* x = lbfgs_malloc(numParameters(params));

	int k = 0;

	if(params.trainPriors)
		for(int i = 0; i < mPriors.size(); ++i, ++k)
			x[k] = mPriors.data()[i];
	if(params.trainScales)
		for(int i = 0; i < mScales.size(); ++i, ++k)
			x[k] = mScales.data()[i];
	if(params.trainWeights)
		for(int i = 0; i < mWeights.size(); ++i, ++k)
			x[k] = mWeights.data()[i];
	if(params.trainFeatures)
		for(int i = 0; i < mFeatures.size(); ++i, ++k)
			x[k] = mFeatures.data()[i];
	if(params.trainCholeskyFactors)
		for(int i = 0; i < mCholeskyFactors.size(); ++i)
			for(int m = 1; m < mDimOut; ++m)
				for(int n = 0; n <= m; ++n, ++k)
					x[k] = mCholeskyFactors[i](m, n);
	if(params.trainPredictors)
		for(int i = 0; i < mPredictors.size(); ++i)
			for(int j = 0; j < mPredictors[i].size(); ++j, ++k)
				x[k] = mPredictors[i].data()[j];
	if(params.trainLinearFeatures)
		for(int i = 0; i < mLinearFeatures.size(); ++i, ++k)
			x[k] = mLinearFeatures.data()[i];
	if(params.trainMeans)
		for(int i = 0; i < mMeans.size(); ++i, ++k)
			x[k] = mMeans.data()[i];

	return x;
}



void CMT::MCGSM::setParameters(const lbfgsfloatval_t* x, const Trainable::Parameters& params_) {
	const Parameters& params = dynamic_cast<const Parameters&>(params_);

	int offset = 0;

	if(params.trainPriors) {
		mPriors = MatrixLBFGS(const_cast<double*>(x), mNumComponents, mNumScales);
		offset += mPriors.size();
	}

	if(params.trainScales) {
		mScales = MatrixLBFGS(const_cast<double*>(x) + offset, mNumComponents, mNumScales);
		offset += mScales.size();
	}

	if(params.trainWeights) {
		mWeights = MatrixLBFGS(const_cast<double*>(x) + offset, mNumComponents, mNumFeatures);
		offset += mWeights.size();
	}

	if(params.trainFeatures) {
		mFeatures = MatrixLBFGS(const_cast<double*>(x) + offset, mDimIn, mNumFeatures);
		offset += mFeatures.size();
	}

	if(params.trainCholeskyFactors)
		for(int i = 0; i < mNumComponents; ++i) {
			mCholeskyFactors[i].setZero();
			mCholeskyFactors[i](0, 0) = 1.;
			for(int m = 1; m < mDimOut; ++m)
				for(int n = 0; n <= m; ++n, ++offset)
					mCholeskyFactors[i](m, n) = x[offset];
		}

	if(params.trainPredictors)
		for(int i = 0; i < mNumComponents; ++i) {
			mPredictors[i] = MatrixLBFGS(const_cast<double*>(x) + offset, mDimOut, mDimIn);
			offset += mPredictors[i].size();
		}

	if(params.trainLinearFeatures) {
		mLinearFeatures = MatrixLBFGS(const_cast<double*>(x) + offset, mNumComponents, mDimIn);
		offset += mLinearFeatures.size();
	}

	if(params.trainMeans) {
		mMeans = MatrixLBFGS(const_cast<double*>(x) + offset, mDimOut, mNumComponents);
		offset += mMeans.size();
	}
}



double CMT::MCGSM::parameterGradient(
	const MatrixXd& inputCompl,
	const MatrixXd& outputCompl,
	const lbfgsfloatval_t* x,
	lbfgsfloatval_t* g,
	const Trainable::Parameters& params_) const
{
	const Parameters& params = dynamic_cast<const Parameters&>(params_);

	// average log-likelihood
	double logLik = 0.;

	// interpret memory for parameters and gradients
	lbfgsfloatval_t* y = const_cast<lbfgsfloatval_t*>(x);

	int offset = 0;

	MatrixLBFGS priors(params.trainPriors ? y : const_cast<double*>(mPriors.data()), mNumComponents, mNumScales);
	MatrixLBFGS priorsGrad(g, mNumComponents, mNumScales);
	if(params.trainPriors)
		offset += priors.size();

	MatrixLBFGS scales(params.trainScales ? y + offset : const_cast<double*>(mScales.data()), mNumComponents, mNumScales);
	MatrixLBFGS scalesGrad(g + offset, mNumComponents, mNumScales);
	if(params.trainScales)
		offset += scales.size();

	MatrixLBFGS weights(params.trainWeights ? y + offset : const_cast<double*>(mWeights.data()), mNumComponents, mNumFeatures);
	MatrixLBFGS weightsGrad(g + offset, mNumComponents, mNumFeatures);
	if(params.trainWeights)
		offset += weights.size();

	MatrixLBFGS features(params.trainFeatures ? y + offset : const_cast<double*>(mFeatures.data()), mDimIn, mNumFeatures);
	MatrixLBFGS featuresGrad(g + offset, mDimIn, mNumFeatures);
	if(params.trainFeatures)
		offset += features.size();

	vector<MatrixXd> choleskyFactors;
	vector<MatrixXd> choleskyFactorsGrad;

	// store memory position of Cholesky factors for later
	int cholFacOffset = offset;

	if(params.trainCholeskyFactors)
		for(int i = 0; i < mNumComponents; ++i) {
			choleskyFactors.push_back(MatrixXd::Zero(mDimOut, mDimOut));
			choleskyFactorsGrad.push_back(MatrixXd::Zero(mDimOut, mDimOut));
			choleskyFactors[i](0, 0) = 1.;
			for(int m = 1; m < mDimOut; ++m)
				for(int n = 0; n <= m; ++n, ++offset)
					choleskyFactors[i](m, n) = x[offset];
		}
	else
		for(int i = 0; i < mNumComponents; ++i)
			choleskyFactors.push_back(mCholeskyFactors[i]);

	vector<MatrixLBFGS> predictors;
	vector<MatrixLBFGS> predictorsGrad;

	if(params.trainPredictors)
		for(int i = 0; i < mNumComponents; ++i) {
			predictors.push_back(MatrixLBFGS(y + offset, mDimOut, mDimIn));
			predictorsGrad.push_back(MatrixLBFGS(g + offset, mDimOut, mDimIn));
			offset += predictors[i].size();
		}
	else
		for(int i = 0; i < mNumComponents; ++i)
			predictors.push_back(MatrixLBFGS(const_cast<double*>(mPredictors[i].data()), mDimOut, mDimIn));

	MatrixLBFGS linearFeatures(params.trainLinearFeatures ? y + offset : const_cast<double*>(mLinearFeatures.data()), mNumComponents, mDimIn);
	MatrixLBFGS linearFeaturesGrad(g + offset, mNumComponents, mDimIn);
	if(params.trainLinearFeatures)
		offset += linearFeatures.size();

	MatrixLBFGS means(params.trainMeans ? y + offset : const_cast<double*>(mMeans.data()), mDimOut, mNumComponents);
	MatrixLBFGS meansGrad(g + offset, mDimOut, mNumComponents);
	if(params.trainMeans)
		offset += means.size();

	if(g) {
		// initialize gradients
		if(params.trainPriors)
			priorsGrad.setZero();
		if(params.trainScales)
			scalesGrad.setZero();
		if(params.trainWeights)
			weightsGrad.setZero();
		if(params.trainFeatures)
			featuresGrad.setZero();
		if(params.trainPredictors)
			for(int i = 0; i < mNumComponents; ++i)
				predictorsGrad[i].setZero();
		if(params.trainLinearFeatures)
			linearFeaturesGrad.setZero();
		if(params.trainMeans)
			meansGrad.setZero();
	}

	// split data into batches for better performance
	int numData = static_cast<int>(inputCompl.cols());
	int batchSize = min(max(params.batchSize, 10), numData);

	for(int b = 0; b < inputCompl.cols(); b += batchSize) {
		const MatrixXd& input = inputCompl.middleCols(b, min(batchSize, numData - b));
		const MatrixXd& output = outputCompl.middleCols(b, min(batchSize, numData - b));

		// compute unnormalized posterior
		MatrixXd featureOutput = features.transpose() * input;
		MatrixXd featureOutputSqr = featureOutput.array().square();
		MatrixXd weightsSqr = weights.array().square();
		MatrixXd weightsOutput = weightsSqr * featureOutputSqr - 2. * linearFeatures * input;

		// containers for intermediate results
		vector<ArrayXXd> logPosteriorIn(mNumComponents);
		vector<ArrayXXd> logPosteriorOut(mNumComponents);
		vector<MatrixXd> predError(mNumComponents);
		vector<Array<double, 1, Dynamic> > predErrorSqNorm(mNumComponents);
		vector<MatrixXd> scalesExp(mNumComponents);

		// partial normalization constants
		ArrayXXd logNormInScales(mNumComponents, input.cols());
		ArrayXXd logNormOutScales(mNumComponents, input.cols());

		#pragma omp parallel for
		for(int i = 0; i < mNumComponents; ++i) {
			scalesExp[i] = scales.row(i).transpose().array().exp();

			MatrixXd negEnergyGate = -scalesExp[i] / 2. * weightsOutput.row(i);
			negEnergyGate.colwise() += priors.row(i).transpose();

			predError[i] = (output - predictors[i] * input).colwise() - means.col(i);
			predErrorSqNorm[i] = (choleskyFactors[i].transpose() * predError[i]).colwise().squaredNorm();

			MatrixXd negEnergyExpert = -scalesExp[i] / 2. * predErrorSqNorm[i].matrix();

			// normalize expert energy
			double logDet = choleskyFactors[i].diagonal().array().abs().log().sum();
			VectorXd logPartf = mDimOut / 2. * scales.row(i).transpose().array()
				+ logDet - mDimOut / 2. * log(2. * PI);

			negEnergyExpert.colwise() += logPartf;

			// unnormalized posterior
			logPosteriorIn[i] = negEnergyGate;
			logPosteriorOut[i] = negEnergyGate + negEnergyExpert;

			// compute normalization constants for posterior over scales
			logNormInScales.row(i) = logSumExp(logPosteriorIn[i]);
			logNormOutScales.row(i) = logSumExp(logPosteriorOut[i]);
		}

		Array<double, 1, Dynamic> logNormIn;
		Array<double, 1, Dynamic> logNormOut;

		// compute normalization constants
		#pragma omp parallel sections
		{
			#pragma omp section
			logNormIn = logSumExp(logNormInScales);
			#pragma omp section
			logNormOut = logSumExp(logNormOutScales);
		}

		// predictive probability
		logLik += (logNormOut - logNormIn).sum();

		if(!g)
			// don't compute gradients
			continue;

		// compute gradients
		#pragma omp parallel for
		for(int i = 0; i < mNumComponents; ++i) {
			// normalize posterior
			logPosteriorIn[i].rowwise() -= logNormIn;
			logPosteriorOut[i].rowwise() -= logNormOut;

			ArrayXXd posteriorIn = logPosteriorIn[i].exp();
			ArrayXXd posteriorOut = logPosteriorOut[i].exp();
			MatrixXd posteriorDiff = posteriorIn - posteriorOut;

			// gradient of prior variables
			if(params.trainPriors)
				priorsGrad.row(i) += posteriorDiff.rowwise().sum();// + 1. * priors;

			Array<double, 1, Dynamic> tmp0 = -scalesExp[i].transpose() * posteriorDiff;

			if(params.trainWeights) {
				Array<double, 1, Dynamic> tmp1 = (featureOutputSqr.array().rowwise() * tmp0).rowwise().sum();

				// gradient of weights
				weightsGrad.row(i) += (tmp1 * weights.row(i).array()).matrix();
			}

			Array<double, 1, Dynamic> tmp3 = posteriorOut.rowwise().sum();

			// gradient of scale variables
			if(params.trainScales) {
				Array<double, 1, Dynamic> tmp2 = (posteriorDiff.array().rowwise() * weightsOutput.row(i).array()).rowwise().sum();
				Array<double, 1, Dynamic> tmp4 = (posteriorOut.rowwise() * predErrorSqNorm[i]).rowwise().sum();

				scalesGrad.row(i) += (
					tmp4 * scales.row(i).array().exp() / 2. -
					tmp3 * mDimOut / 2. -
					tmp2 * scales.row(i).array().exp() / 2.).matrix();
			}

			// partial gradient of features
			if(params.trainFeatures) {
				MatrixXd tmp5 = input.array().rowwise() * tmp0;
				ArrayXXd tmp6 = tmp5 * featureOutput.transpose();

				#pragma omp critical
				featuresGrad += (tmp6.rowwise() * weightsSqr.row(i).array()).matrix();
			}

			Array<double, 1, Dynamic> tmp7 = scalesExp[i].transpose() * posteriorOut.matrix();
			MatrixXd precision = choleskyFactors[i] * choleskyFactors[i].transpose();
			MatrixXd tmp8 = predError[i].array().rowwise() * tmp7;
			MatrixXd tmp9 = precision * tmp8;

			// gradient of cholesky factor
			if(params.trainCholeskyFactors) {
				MatrixXd tmp10 = choleskyFactors[i].diagonal().cwiseInverse().asDiagonal();
				choleskyFactorsGrad[i] += tmp8 * predError[i].transpose() * choleskyFactors[i]
					- tmp3.sum() * tmp10;
			}

			// gradient of linear predictor
			if(params.trainPredictors)
				predictorsGrad[i] -= tmp9 * input.transpose();

			if(params.trainLinearFeatures)
				linearFeaturesGrad.row(i) -= (tmp0.matrix() * input.transpose()).colwise().sum();

			if(params.trainMeans)
				meansGrad.col(i) -= tmp9.rowwise().sum();
		}
	}

	double normConst = inputCompl.cols() * log(2.) * dimOut();

	if(g) {
		// write back gradients of Cholesky factors
		if(params.trainCholeskyFactors)
			for(int i = 0; i < mNumComponents; ++i)
				for(int m = 1; m < mDimOut; ++m)
					for(int n = 0; n <= m; ++n, ++cholFacOffset)
						g[cholFacOffset] = choleskyFactorsGrad[i](m, n);

		// normalize gradient by number of data points
		for(int i = 0; i < offset; ++i)
			g[i] /= normConst;

		// regularization
		if(params.trainFeatures)
			featuresGrad += params.regularizeFeatures.gradient(features);

		if(params.trainWeights)
			weightsGrad += params.regularizeWeights.gradient(weights);

		if(params.trainPredictors)
			#pragma omp parallel for
			for(int i = 0; i < mNumComponents; ++i)
				predictorsGrad[i] += params.regularizePredictors.gradient(predictors[i].transpose()).transpose();

		if(params.trainLinearFeatures)
			linearFeaturesGrad += params.regularizeLinearFeatures.gradient(linearFeatures.transpose()).transpose();

		if(params.trainMeans)
			meansGrad += params.regularizeMeans.gradient(means);
	}

	double value = -logLik / normConst;

	// regularization
	if(params.trainFeatures)
		value += params.regularizeFeatures.evaluate(features);

	if(params.trainWeights)
		value += params.regularizeWeights.evaluate(weights);

	if(params.trainPredictors)
		for(int i = 0; i < mNumComponents; ++i)
			value += params.regularizePredictors.evaluate(predictors[i].transpose());

	if(params.trainLinearFeatures)
		value += params.regularizeLinearFeatures.evaluate(linearFeatures.transpose());

	if(params.trainMeans)
		value += params.regularizeMeans.evaluate(means);

	// return negative penalized average log-likelihood
	return value;
}



pair<pair<ArrayXXd, ArrayXXd>, Array<double, 1, Dynamic> > CMT::MCGSM::computeDataGradient(
	const MatrixXd& input,
	const MatrixXd& output) const
{
	// compute unnormalized posterior
	MatrixXd featureOutput = mFeatures.transpose() * input;
	MatrixXd featureOutputSqr = featureOutput.array().square();
	MatrixXd weightsSqr = mWeights.array().square();
	MatrixXd weightsSqrOutput = weightsSqr * featureOutputSqr - 2. * mLinearFeatures * input;

	// containers for intermediate results
	vector<ArrayXXd> logPosteriorIn(mNumComponents);
	vector<ArrayXXd> logPosteriorOut(mNumComponents);
	vector<MatrixXd> predError(mNumComponents);
	vector<VectorXd> scalesExp(mNumComponents);

	// partial normalization constants
	ArrayXXd logNormInScales(mNumComponents, input.cols());
	ArrayXXd logNormOutScales(mNumComponents, input.cols());

	#pragma omp parallel for
	for(int i = 0; i < mNumComponents; ++i) {
		scalesExp[i] = mScales.row(i).transpose().array().exp();

		// gate energy
		ArrayXXd negEnergyGate = -scalesExp[i] / 2. * weightsSqrOutput.row(i);
		negEnergyGate.colwise() += mPriors.row(i).transpose();

		predError[i] = mCholeskyFactors[i].transpose() * ((output - mPredictors[i] * input).colwise() - mMeans.col(i));
		ArrayXXd negEnergyExpert = -scalesExp[i] / 2. * predError[i].colwise().squaredNorm();

		// normalize expert energy
		double logDet = mCholeskyFactors[i].diagonal().array().abs().log().sum();
		negEnergyExpert.colwise() += mDimOut / 2. * mScales.row(i).transpose() 
			+ logDet - mDimOut / 2. * log(2. * PI);

		// unnormalized posterior
		logPosteriorIn[i] = negEnergyGate;
		logPosteriorOut[i] = negEnergyGate + negEnergyExpert;

		// compute normalization constants for posterior over scales
		logNormInScales.row(i) = logSumExp(logPosteriorIn[i]);
		logNormOutScales.row(i) = logSumExp(logPosteriorOut[i]);
	}

	Array<double, 1, Dynamic> logNormIn;
	Array<double, 1, Dynamic> logNormOut;

	// compute normalization constants
	#pragma omp parallel sections
	{
		#pragma omp section
		logNormIn = logSumExp(logNormInScales);
		#pragma omp section
		logNormOut = logSumExp(logNormOutScales);
	}

	Array<double, 1, Dynamic> logLikelihood = logNormOut - logNormIn;

	ArrayXXd inputGradients = ArrayXXd::Zero(mDimIn, input.cols());
	ArrayXXd outputGradients = ArrayXXd::Zero(mDimOut, output.cols());

	#pragma omp parallel for
	for(int i = 0; i < mNumComponents; ++i) {
		// normalize posterior
		logPosteriorIn[i].rowwise() -= logNormIn;
		logPosteriorOut[i].rowwise() -= logNormOut;

		// posterior over this component and all scales
		MatrixXd posteriorIn = logPosteriorIn[i].exp();
		MatrixXd posteriorOut = logPosteriorOut[i].exp();
		MatrixXd posteriorDiff = posteriorOut - posteriorIn;

		ArrayXXd dpdy = -mCholeskyFactors[i] * predError[i];
		ArrayXXd dpdx = -mPredictors[i].transpose() * dpdy.matrix();
		ArrayXXd dfdx = -(mFeatures.array().rowwise() * weightsSqr.row(i).array()).matrix() * featureOutput;
		dfdx.colwise() += mLinearFeatures.col(i).array();

		// weights for this component
		Array<double, 1, Dynamic> weightsOut = scalesExp[i].transpose() * posteriorOut;
		Array<double, 1, Dynamic> weightsDiff = scalesExp[i].transpose() * posteriorDiff;

		#pragma omp critical
		{
			inputGradients += dpdx.rowwise() * weightsOut + dfdx.rowwise() * weightsDiff;
			outputGradients += dpdy.rowwise() * weightsOut;
		}
	}

	return make_pair(make_pair(inputGradients, outputGradients), logLikelihood);
}



bool CMT::MCGSM::train(
	const MatrixXd& input,
	const MatrixXd& output,
	const MatrixXd* inputVal,
	const MatrixXd* outputVal,
	const Trainable::Parameters& params_)
{
	if(!mDimIn) {
		const Parameters& params = dynamic_cast<const Parameters&>(params_);

		// MCGSM reduces to MoGSM for zero-dimensional inputs
		MoGSM mogsm(mDimOut, mNumComponents, mNumScales);

		// initialize model
		ArrayXd priors = mPriors.exp().rowwise().sum();
		mogsm.setPriors(priors / priors.sum());

		for(int k = 0; k < mNumComponents; ++k) {
			ArrayXd priors = mPriors.row(k).exp().transpose();
			GSM* gsm = dynamic_cast<GSM*>(mogsm[k]);
			gsm->setMean(mMeans.col(k));
			gsm->setPriors(priors / priors.sum());
			gsm->setScales(mScales.row(k).exp().transpose());
			gsm->setCholesky(mCholeskyFactors[k]);
		}

		// optimization hyperparameters
		MoGSM::Parameters mogsmParams;
		mogsmParams.initialize = false;
		mogsmParams.verbosity = params.verbosity;
		mogsmParams.maxIter = params.maxIter;
		mogsmParams.threshold = params.threshold;
		mogsmParams.valIter = params.valIter;
		mogsmParams.valLookAhead = params.valLookAhead;
		mogsmParams.trainPriors = params.trainPriors;

		MoGSM::Component::Parameters gsmParams;
		gsmParams.trainMean = params.trainMeans;
		gsmParams.trainPriors = params.trainPriors;
		gsmParams.trainCovariance = params.trainCholeskyFactors;
		gsmParams.trainScales = params.trainScales;

		// fit parameters of model to data
		bool converged;
		if(outputVal)
			converged = mogsm.train(output, *outputVal, mogsmParams, gsmParams);
		else
			converged = mogsm.train(output, mogsmParams, gsmParams);

		// copy parameters back
		mPriors.colwise() = mogsm.priors().array().log();

		vector<MatrixXd> choleskyFactors;

		for(int k = 0; k < mNumComponents; ++k) {
			GSM* gsm = dynamic_cast<GSM*>(mogsm[k]);

			mPriors.row(k) += gsm->priors().array().log().transpose();
			mScales.row(k) = gsm->scales().array().log().transpose();
			choleskyFactors.push_back(gsm->cholesky());
			mMeans.col(k) = gsm->mean();
		}

		setCholeskyFactors(choleskyFactors);

		return converged;
	} else {
		return Trainable::train(input, output, inputVal, outputVal, params_);
	}
}
