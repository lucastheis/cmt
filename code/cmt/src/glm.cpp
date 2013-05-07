#include "exception.h"

#include "glm.h"
using CMT::GLM;
using CMT::LogisticFunction;
using CMT::Bernoulli;

#include <cmath>
using std::log;

#include <map>
using std::pair;
using std::make_pair;

#include "Eigen/Core"
using Eigen::Dynamic;
using Eigen::Array;
using Eigen::ArrayXXd;
using Eigen::MatrixXd;

GLM::Nonlinearity* defaultNonlinearity = new LogisticFunction;
GLM::UnivariateDistribution* defaultDistribution = new Bernoulli;

#include <iostream>

CMT::GLM::GLM(
	int dimIn,
	Nonlinearity* nonlinearity,
	UnivariateDistribution* distribution) :
	mDimIn(dimIn),
	mNonlinearity(nonlinearity),
	mDistribution(distribution)
{
	if(mDimIn < 0)
		throw Exception("Input dimensionality should be non-negative.");
	if(!mNonlinearity)
		throw Exception("No nonlinearity specified.");
	if(!mDistribution)
		throw Exception("No distribution specified.");

	mWeights = VectorXd::Random(dimIn) / 100.;
}



CMT::GLM::GLM(int dimIn) : mDimIn(dimIn) {
	if(mDimIn < 0)
		throw Exception("Input dimensionality should be non-negative.");

	mNonlinearity = defaultNonlinearity;
	mDistribution = defaultDistribution;

	mWeights = VectorXd::Random(dimIn) / 100.;
}




CMT::GLM::GLM(int dimIn, const GLM& glm) : 
	mDimIn(dimIn),
	mNonlinearity(glm.mNonlinearity),
	mDistribution(glm.mDistribution)
{
	if(mDimIn < 0)
		throw Exception("Input dimensionality should be non-negative.");

	mWeights = VectorXd::Random(dimIn) / 100.;
}



CMT::GLM::~GLM() {
}



Array<double, 1, Dynamic> CMT::GLM::logLikelihood(
	const MatrixXd& input,
	const MatrixXd& output) const
{
	if(input.rows() != mDimIn)
		throw Exception("Input has wrong dimensionality.");

	if(!mDimIn)
		mDistribution->logLikelihood(output, ArrayXXd::Zero(1, input.cols()));

	return mDistribution->logLikelihood(
		output,
		(*mNonlinearity)(mWeights.transpose() * input));
}



MatrixXd CMT::GLM::sample(const MatrixXd& input) const {
	if(input.rows() != mDimIn)
		throw Exception("Input has wrong dimensionality.");

	if(!mDimIn)
		return mDistribution->sample(ArrayXXd::Zero(1, input.cols()));

	return mDistribution->sample((*mNonlinearity)(mWeights * input));
}



int CMT::GLM::numParameters(const Parameters& params) const {
	return mWeights.size();
}



lbfgsfloatval_t* CMT::GLM::parameters(const Parameters& params) const {
	lbfgsfloatval_t* x = lbfgs_malloc(mWeights.size());

	for(int i = 0; i < mWeights.size(); ++i)
		x[i] = mWeights.data()[i];

	return x;
}



void CMT::GLM::setParameters(const lbfgsfloatval_t* x, const Parameters& params) {
	for(int i = 0; i < mWeights.size(); ++i)
		mWeights.data()[i] = x[i];
}



double CMT::GLM::parameterGradient(
	const MatrixXd& input,
	const MatrixXd& output,
	const lbfgsfloatval_t* x,
	lbfgsfloatval_t* g,
	const Parameters& params) const
{
	VectorLBFGS weights(const_cast<lbfgsfloatval_t*>(x), mWeights.size());
	VectorLBFGS weightsGrad(g, mWeights.size());

	Array<double, 1, Dynamic> responses = weights.transpose() * input;
	Array<double, 1, Dynamic> means = (*mNonlinearity)(responses);

	if(g) {
		Array<double, 1, Dynamic> tmp1 = mDistribution->gradient(output, means);
		Array<double, 1, Dynamic> tmp2 = mNonlinearity->derivative(responses);
		weightsGrad = (input.array().rowwise() * (tmp1 * tmp2)).rowwise().mean() / log(2.);
	}

	return -mDistribution->logLikelihood(output, means).mean() / log(2.);
}



pair<pair<ArrayXXd, ArrayXXd>, Array<double, 1, Dynamic> > CMT::GLM::computeDataGradient(
	const MatrixXd& input,
	const MatrixXd& output) const
{
	return make_pair(
		make_pair(
			ArrayXXd::Zero(input.rows(), input.cols()), 
			ArrayXXd::Zero(output.rows(), output.cols())), 
		logLikelihood(input, output));
}



bool CMT::GLM::train(
	const MatrixXd& input,
	const MatrixXd& output,
	const MatrixXd* inputVal,
	const MatrixXd* outputVal,
	const Trainable::Parameters& params)
{
	if(!mDimIn)
		return true;
	else
		return Trainable::train(input, output, inputVal, outputVal, params);
}



ArrayXXd CMT::LogisticFunction::operator()(const ArrayXXd& data) const {
	return 1. / (1. + (-data).exp());
}



ArrayXXd CMT::LogisticFunction::derivative(const ArrayXXd& data) const {
	ArrayXXd tmp = operator()(data);
	return tmp * (1. - tmp);
}



CMT::Bernoulli::Bernoulli(double prob) : mProb(prob) {
	if(prob < 0. || prob > 1.)
		throw Exception("Probability has to be between 0 and 1.");
}



MatrixXd CMT::Bernoulli::sample(int numSamples) const {
	return (Array<double, 1, Dynamic>::Random(numSamples).abs() < mProb).cast<double>();
}



MatrixXd CMT::Bernoulli::sample(const Array<double, 1, Dynamic>& means) const {
	return (Array<double, 1, Dynamic>::Random(means.size()).abs() < means).cast<double>();
}



Array<double, 1, Dynamic> CMT::Bernoulli::logLikelihood(const MatrixXd& data) const {
	if(mProb > 0. && 1. - mProb > 0.)
		return log(mProb) * data.array() + log(1. - mProb) * (1. - data.array());

	Array<double, 1, Dynamic> logLik = Array<double, 1, Dynamic>(data.size());

	double logProb1 = log(mProb);
	double logProb0 = log(1. - mProb);

	for(int i = 0; i < data.size(); ++i)
		logLik[i] = data.data()[i] > 0.5 ? logProb1 : logProb0;

	return logLik;
}



Array<double, 1, Dynamic> CMT::Bernoulli::logLikelihood(
	const Array<double, 1, Dynamic>& data,
	const Array<double, 1, Dynamic>& means) const
{
	Array<double, 1, Dynamic> logLik = Array<double, 1, Dynamic>(data.size());

	for(int i = 0; i < data.size(); ++i)
		logLik[i] = data[i] > 0.5 ? log(means[i]) : log(1. - means[i]);

	return logLik;
}



Array<double, 1, Dynamic> CMT::Bernoulli::gradient(
	const Array<double, 1, Dynamic>& data,
	const Array<double, 1, Dynamic>& means) const
{
	Array<double, 1, Dynamic> grad = Array<double, 1, Dynamic>(data.size());

	for(int i = 0; i < data.size(); ++i)
		grad[i] = data[i] > 0.5 ? -1. / means[i] : 1. / (1. - means[i]);

	return grad;
}
