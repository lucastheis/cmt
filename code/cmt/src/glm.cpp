#include "exception.h"

#include "glm.h"
using CMT::GLM;

#include "nonlinearities.h"
using CMT::Nonlinearity;
using CMT::LogisticFunction;

#include "univariatedistributions.h"
using CMT::UnivariateDistribution;
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

Nonlinearity* const GLM::defaultNonlinearity = new LogisticFunction;
UnivariateDistribution* const GLM::defaultDistribution = new Bernoulli;

CMT::GLM::GLM(
	int dimIn,
	Nonlinearity* nonlinearity,
	UnivariateDistribution* distribution) :
	mDimIn(dimIn),
	mNonlinearity(nonlinearity ? nonlinearity : defaultNonlinearity),
	mDistribution(distribution ? distribution : defaultDistribution)
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

	Array<double, 1, Dynamic> responses;

	if(mDimIn)
		responses = (mWeights.transpose() * input).array() + mBias;
	else
		responses = Array<double, 1, Dynamic>::Constant(output.cols(), mBias);

	return mDistribution->logLikelihood(output, (*mNonlinearity)(responses));
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

 	// check if nonlinearity is differentiable
 	DifferentiableNonlinearity* nonlinearity = dynamic_cast<DifferentiableNonlinearity*>(mNonlinearity);

	if(!nonlinearity)
		throw Exception("Nonlinearity has to be differentiable for training.");
 
 	// initialize gradient and log-likelihood
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
 			Array<double, 1, Dynamic> tmp2 = nonlinearity->derivative(responses);
 			Array<double, 1, Dynamic> tmp3 = tmp1 * tmp2;
 
 			// weights gradient
 			if(mDimIn) {
 				VectorXd weightsGrad_ = (input.array().rowwise() * tmp3).rowwise().sum();
 
 				#pragma omp critical
 				weightsGrad += weightsGrad_;
 			}
 
 			// bias gradient
 			#pragma omp critical
 			g[mDimIn] += tmp3.sum();
 		}
 
 		#pragma omp critical
 		logLik -= mDistribution->logLikelihood(output, means).sum();
 	}

 	double normConst = outputCompl.cols() * log(2.);

 	if(g)
		for(int i = 0; i <= mDimIn; ++i)
			g[i] /= normConst;
 	
 	return logLik / normConst;
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
