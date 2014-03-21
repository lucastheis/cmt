#include "mlr.h"
#include "utils.h"

#include "Eigen/Core"
using Eigen::Array;
using Eigen::ArrayXXd;
using Eigen::MatrixXd;
using Eigen::Dynamic;
using Eigen::RowMajor;

#include <cstdlib>
using std::rand;

#include <cmath>
using std::log;

#include <map>
using std::pair;
using std::make_pair;

#include <iostream>

CMT::MLR::Parameters::Parameters() :
	Trainable::Parameters::Parameters(),
	trainWeights(true),
	trainBiases(true),
	regularizeWeights(0.),
	regularizeBiases(0.)
{
}



CMT::MLR::Parameters::Parameters(const Parameters& params) :
	Trainable::Parameters::Parameters(params),
	trainWeights(params.trainWeights),
	trainBiases(params.trainBiases),
	regularizeWeights(params.regularizeWeights),
	regularizeBiases(params.regularizeBiases)
{
}



CMT::MLR::Parameters& CMT::MLR::Parameters::operator=(const Parameters& params) {
	Trainable::Parameters::operator=(params);

	trainWeights = params.trainWeights;
	trainBiases = params.trainBiases;
	regularizeWeights = params.regularizeWeights;
	regularizeBiases = params.regularizeBiases;

	return *this;
}



CMT::MLR::MLR(int dimIn, int dimOut) : 
	mDimIn(dimIn),
	mDimOut(dimOut),
	mWeights(MatrixXd::Zero(dimOut, dimIn)),
	mBiases(VectorXd::Zero(dimOut))
{
	if(dimIn < 1)
		throw Exception("Input dimensionality has to be positive.");
	if(dimOut < 1)
		throw Exception("Output dimensionality has to be positive.");
}



CMT::MLR::~MLR() {
}



Array<double, 1, Dynamic> CMT::MLR::logLikelihood(
	const MatrixXd& input,
	const MatrixXd& output) const
{
	if(input.cols() != output.cols())
		throw Exception("Number of inputs and outputs have to be the same.");
	if(input.rows() != mDimIn)
		throw Exception("Inputs have wrong dimensionality.");
	if(output.rows() != mDimOut)
		throw Exception("Output has wrong dimensionality.");

	// distribution over outputs
	ArrayXXd logProb = (mWeights * input).colwise() + mBiases;
	logProb.rowwise() -= logSumExp(logProb);

	return (logProb * output.array()).colwise().sum();
}



MatrixXd CMT::MLR::sample(const MatrixXd& input) const {
	if(input.rows() != mDimIn)
		throw Exception("Inputs have wrong dimensionality.");

	// distribution over outputs
	ArrayXXd prob = predict(input);

	MatrixXd output = MatrixXd::Zero(mDimOut, input.cols());

	#pragma omp parallel for
	for(int j = 0; j < input.cols(); ++j) {
		double urand = static_cast<double>(rand()) / RAND_MAX;
		double cdf = 0.;

		for(int k = 0; k < mDimOut; ++k) {
			cdf += prob(k, j);

			if(urand < cdf) {
				output(k, j) = 1.;
				break;
			}
		}
	}

	return output;
}



MatrixXd CMT::MLR::predict(const MatrixXd& input) const {
	if(input.rows() != mDimIn)
		throw Exception("Inputs have wrong dimensionality.");

	MatrixXd output = MatrixXd::Zero(mDimOut, input.cols());

	// distribution over outputs
	ArrayXXd prob = (mWeights * input).colwise() + mBiases;
	prob.rowwise() -= logSumExp(prob);
	prob = prob.exp();

	return prob;
}



pair<pair<ArrayXXd, ArrayXXd>, Array<double, 1, Dynamic> > CMT::MLR::computeDataGradient(
	const MatrixXd& input,
	const MatrixXd& output) const
{
	return make_pair(
		make_pair(
			ArrayXXd::Zero(input.rows(), input.cols()), 
			ArrayXXd::Zero(output.rows(), output.cols())), 
		logLikelihood(input, output));
}



int CMT::MLR::numParameters(const Trainable::Parameters& params_) const {
	const Parameters& params = dynamic_cast<const Parameters&>(params_);

	int numParameters = 0;

	if(params.trainWeights)
		numParameters += mDimIn * (mDimOut - 1);
	if(params.trainBiases)
		numParameters += mDimOut - 1;

	return numParameters;
}



lbfgsfloatval_t* CMT::MLR::parameters(const Trainable::Parameters& params_) const {
	const Parameters& params = dynamic_cast<const Parameters&>(params_);

	lbfgsfloatval_t* x = lbfgs_malloc(numParameters(params));

	int k = 0;

	// copy parameters
	if(params.trainWeights)
		for(int i = 1; i < mWeights.rows(); ++i)
			for(int j = 0; j < mWeights.cols(); ++j, ++k)
				x[k] = mWeights(i, j);
	if(params.trainBiases)
		for(int i = 1; i < mBiases.rows(); ++i, ++k)
			x[k] = mBiases[i];

	return x;
}



void CMT::MLR::setParameters(
	const lbfgsfloatval_t* x,
	const Trainable::Parameters& params_)
{
	const Parameters& params = dynamic_cast<const Parameters&>(params_);

	int k = 0;

	// copy parameters
	if(params.trainWeights)
		for(int i = 1; i < mWeights.rows(); ++i)
			for(int j = 0; j < mWeights.cols(); ++j, ++k)
				mWeights(i, j) = x[k];
	if(params.trainBiases)
		for(int i = 1; i < mBiases.rows(); ++i, ++k)
			mBiases[i] = x[k];
}



double CMT::MLR::parameterGradient(
	const MatrixXd& input,
	const MatrixXd& output,
	const lbfgsfloatval_t* x,
	lbfgsfloatval_t* g,
	const Trainable::Parameters& params_) const
{
	const Parameters& params = dynamic_cast<const Parameters&>(params_);

	MatrixXd weights = mWeights;
	VectorXd biases = mBiases;

	// copy parameters
	int k = 0;
	if(params.trainWeights)
		for(int i = 1; i < weights.rows(); ++i)
			for(int j = 0; j < weights.cols(); ++j, ++k)
				weights(i, j) = x[k];
	if(params.trainBiases)
		for(int i = 1; i < mBiases.rows(); ++i, ++k)
			biases[i] = x[k];

	// compute distribution over outputs
	ArrayXXd logProb = (weights * input).colwise() + biases;
	logProb.rowwise() -= logSumExp(logProb);

	// difference between prediction and actual output
	MatrixXd diff = (logProb.exp().matrix() - output);

	// compute gradients
	double normConst = output.cols() * log(2.);

	if(g) {
		int offset = 0;

		if(params.trainWeights) {
			Map<Matrix<double, Dynamic, Dynamic, RowMajor> > weightsGrad(g, mDimOut - 1, mDimIn);
			weightsGrad = (diff * input.transpose() / normConst).bottomRows(mDimOut - 1);
			offset += weightsGrad.size();

			weightsGrad += params.regularizeWeights.gradient(
				weights.bottomRows(mDimOut - 1).transpose()).transpose();
		}

		if(params.trainBiases) {
			VectorLBFGS biasesGrad(g + offset, mDimOut - 1);
			biasesGrad = diff.rowwise().sum().bottomRows(mDimOut - 1) / normConst;
			biasesGrad += params.regularizeBiases.gradient(biases);
		}
	}

	// return negative average log-likelihood in bits
	double value = -(logProb * output.array()).sum() / normConst;

	if(params.trainWeights)
		value += params.regularizeWeights.evaluate(weights.bottomRows(mDimOut - 1).transpose());

	if(params.trainBiases)
		value += params.regularizeBiases.evaluate(biases);

	return value;
}



double CMT::MLR::evaluate(
	const MatrixXd& input,
	const MatrixXd& output) const
{
	return -logLikelihood(input, output).mean() / log(2.);
}



double CMT::MLR::evaluate(
	const MatrixXd& input,
	const MatrixXd& output,
	const Preconditioner& preconditioner) const
{
	return -logLikelihood(preconditioner(input, output)).mean() / log(2.)
		- preconditioner.logJacobian(input, output).mean() / log(2.);
}



double CMT::MLR::evaluate(
	const pair<ArrayXXd, ArrayXXd>& data) const
{
	return -logLikelihood(data.first, data.second).mean() / log(2.);
}



double CMT::MLR::evaluate(
	const pair<ArrayXXd, ArrayXXd>& data,
	const Preconditioner& preconditioner) const
{
	return -logLikelihood(preconditioner(data.first, data.second)).mean() / log(2.)
		- preconditioner.logJacobian(data).mean() / log(2.);
}
