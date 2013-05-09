#include "exception.h"

#include "glm.h"
using CMT::GLM;
using CMT::LogisticFunction;
using CMT::Bernoulli;

#include <cmath>
using std::log;
using std::min;

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
	mBias = 0.;
}



CMT::GLM::GLM(int dimIn) : mDimIn(dimIn) {
	if(mDimIn < 0)
		throw Exception("Input dimensionality should be non-negative.");

	mNonlinearity = defaultNonlinearity;
	mDistribution = defaultDistribution;

	mWeights = VectorXd::Random(dimIn) / 100.;
	mBias = 0.;
}




CMT::GLM::GLM(int dimIn, const GLM& glm) : 
	mDimIn(dimIn),
	mNonlinearity(glm.mNonlinearity),
	mDistribution(glm.mDistribution)
{
	if(mDimIn < 0)
		throw Exception("Input dimensionality should be non-negative.");

	mWeights = VectorXd::Random(dimIn) / 100.;
	mBias = 0.;
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
		return mDistribution->logLikelihood(
			output, Array<double, 1, Dynamic>::Constant(output.cols(), mBias));

	return mDistribution->logLikelihood(
		output,
		(*mNonlinearity)((mWeights.transpose() * input).array() + mBias));
}



MatrixXd CMT::GLM::sample(const MatrixXd& input) const {
	if(input.rows() != mDimIn)
		throw Exception("Input has wrong dimensionality.");

	if(!mDimIn)
		return mDistribution->sample(Array<double, 1, Dynamic>::Constant(input.cols(), mBias));

	return mDistribution->sample((*mNonlinearity)((mWeights.transpose() * input).array() + mBias));
}



int CMT::GLM::numParameters(const Parameters& params) const {
	return mDimIn + 1;
}



lbfgsfloatval_t* CMT::GLM::parameters(const Parameters& params) const {
	lbfgsfloatval_t* x = lbfgs_malloc(mDimIn + 1);

	for(int i = 0; i < mDimIn; ++i)
		x[i] = mWeights.data()[i];
	x[mDimIn] = mBias;

	return x;
}



void CMT::GLM::setParameters(const lbfgsfloatval_t* x, const Parameters& params) {
	for(int i = 0; i < mDimIn; ++i)
		mWeights.data()[i] = x[i];
	mBias = x[mDimIn];
}



double CMT::GLM::parameterGradient(
	const MatrixXd& inputCompl,
	const MatrixXd& outputCompl,
	const lbfgsfloatval_t* x,
	lbfgsfloatval_t* g,
	const Parameters& params) const
{
  	int numData = static_cast<int>(inputCompl.cols());
 	int batchSize = min(params.batchSize, numData);
 
 	VectorLBFGS weights(const_cast<lbfgsfloatval_t*>(x), mDimIn);
 	VectorLBFGS weightsGrad(g, mDimIn);
 	double bias = x[mDimIn];
 
 	if(g) {
 		weightsGrad.setZero();
 		g[mDimIn] = 0.;
	}
 	double logLik = 0.;
 
 	#pragma omp parallel for
 	for(int b = 0; b < inputCompl.cols(); b += batchSize) {
 		const MatrixXd& input = inputCompl.middleCols(b, min(batchSize, numData - b));
 		const MatrixXd& output = outputCompl.middleCols(b, min(batchSize, numData - b));
 
 		// linear responses
 		Array<double, 1, Dynamic> responses;
 
 		if(mDimIn)
 			responses = (weights.transpose() * input).array() + bias;
 		else
 			responses = Array<double, 1, Dynamic>::Constant(output.cols(), bias);
 
 		// nonlinear responses
 		Array<double, 1, Dynamic> means = (*mNonlinearity)(responses);
 
 		if(g) {
 			Array<double, 1, Dynamic> tmp1 = mDistribution->gradient(output, means);
 			Array<double, 1, Dynamic> tmp2 = mNonlinearity->derivative(responses);
 			Array<double, 1, Dynamic> tmp3 = tmp1 * tmp2;
 
 			// weights gradient
 			if(mDimIn) {
 				VectorXd weightsGrad_ = (input.array().rowwise() * tmp3).rowwise().mean() / log(2.);
 
 				#pragma omp critical
 				weightsGrad += weightsGrad_;
 			}
 
 			// bias gradient
 			#pragma omp critical
 			g[mDimIn] += tmp3.mean() / log(2.);
 		}
 
 		#pragma omp critical
 		logLik -= mDistribution->logLikelihood(output, means).mean() / log(2.);
 	}
 	
 	return logLik;
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



CMT::GLM::Nonlinearity::~Nonlinearity() {
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
