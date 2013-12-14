#include "exception.h"
#include "utils.h"

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

CMT::GLM::Parameters::Parameters() :
	Trainable::Parameters::Parameters(),
	trainWeights(true),
	trainBias(true),
	trainNonlinearity(false),
	regularizeWeights(0.),
	regularizeBias(0.),
	regularizer(L2)
{
}



CMT::GLM::Parameters::Parameters(const Parameters& params) :
	Trainable::Parameters::Parameters(params),
	trainWeights(params.trainWeights),
	trainBias(params.trainBias),
	trainNonlinearity(params.trainNonlinearity),
	regularizeWeights(params.regularizeWeights),
	regularizeBias(params.regularizeBias),
	regularizer(params.regularizer)
{
}



CMT::GLM::Parameters& CMT::GLM::Parameters::operator=(const Parameters& params) {
	Trainable::Parameters::operator=(params);

	trainWeights = params.trainWeights;
	trainBias = params.trainBias;
	trainNonlinearity = params.trainNonlinearity;
	regularizeWeights = params.regularizeWeights;
	regularizeBias = params.regularizeBias;
	regularizer = params.regularizer;

	return *this;
}



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



int CMT::GLM::numParameters(const Trainable::Parameters& params_) const {
	const Parameters& params = dynamic_cast<const Parameters&>(params_);
	
	int numParams = 0;

	if(params.trainWeights)
		numParams += mDimIn;
	if(params.trainBias)
		numParams += 1;
	if(params.trainNonlinearity) {
		// test if nonlinearity is trainable
		TrainableNonlinearity* nonlinearity =
			dynamic_cast<TrainableNonlinearity*>(mNonlinearity);
		if(!nonlinearity)
			throw Exception("Nonlinearity has to be trainable.");
		numParams += nonlinearity->numParameters();
	}

	return numParams;
}



lbfgsfloatval_t* CMT::GLM::parameters(const Trainable::Parameters& params_) const {
	const Parameters& params = dynamic_cast<const Parameters&>(params_);

	lbfgsfloatval_t* x = lbfgs_malloc(numParameters(params));

	int k = 0;

	if(params.trainWeights)
		for(int i = 0; i < mDimIn; ++i, ++k)
			x[k] = mWeights[i];
	if(params.trainBias)
		x[k++] = mBias;
	if(params.trainNonlinearity) {
		// test if nonlinearity is trainable
		TrainableNonlinearity* nonlinearity =
			dynamic_cast<TrainableNonlinearity*>(mNonlinearity);
		if(!nonlinearity)
			throw Exception("Nonlinearity has to be trainable.");

		ArrayXd nonlParams = nonlinearity->parameters();

		for(int i = 0; i < nonlParams.size(); ++i, ++k)
			x[k] = nonlParams[i];
	}

	return x;
}



void CMT::GLM::setParameters(
	const lbfgsfloatval_t* x,
	const Trainable::Parameters& params_)
{
	const Parameters& params = dynamic_cast<const Parameters&>(params_);

	int k = 0;

	if(params.trainWeights)
		for(int i = 0; i < mDimIn; ++i, ++k)
			mWeights[i] = x[k];
	if(params.trainBias)
		mBias = x[k++];
	if(params.trainNonlinearity) {
		// test if nonlinearity is trainable
		TrainableNonlinearity* nonlinearity =
			dynamic_cast<TrainableNonlinearity*>(mNonlinearity);
		if(!nonlinearity)
			throw Exception("Nonlinearity has to be trainable.");

		ArrayXd nonlParams(nonlinearity->numParameters());
		for(int i = 0; i < nonlParams.size(); ++i, ++k)
			nonlParams[i] = x[k];

		nonlinearity->setParameters(nonlParams);
	}
}



double CMT::GLM::parameterGradient(
	const MatrixXd& inputCompl,
	const MatrixXd& outputCompl,
	const lbfgsfloatval_t* x,
	lbfgsfloatval_t* g,
	const Trainable::Parameters& params_) const
{
	const Parameters& params = dynamic_cast<const Parameters&>(params_);

	// check if nonlinearity is trainable and/or differentiable
	TrainableNonlinearity* trainableNonlinearity = 
		dynamic_cast<TrainableNonlinearity*>(mNonlinearity);
	DifferentiableNonlinearity* differentiableNonlinearity =
		dynamic_cast<DifferentiableNonlinearity*>(mNonlinearity);

	if((params.trainWeights || params.trainBias) && !differentiableNonlinearity)
		throw Exception("Nonlinearity has to be differentiable.");
	if(params.trainNonlinearity && !trainableNonlinearity)
		throw Exception("Nonlinearity is not trainable.");

	int numData = static_cast<int>(inputCompl.cols());
	int batchSize = min(params.batchSize, numData);

	lbfgsfloatval_t* y = const_cast<lbfgsfloatval_t*>(x);
	int offset = 0;

	// interpret parameters
	VectorLBFGS weights(params.trainWeights ? y : const_cast<double*>(mWeights.data()), mDimIn);
	VectorLBFGS weightsGrad(g, mDimIn);
	if(params.trainWeights)
		offset += weights.size();

	double bias = params.trainBias ? y[offset] : mBias;
	double* biasGrad = g + offset;
	if(params.trainBias)
		offset += 1;

	VectorLBFGS nonlinearityGrad(g + offset,
		trainableNonlinearity ? trainableNonlinearity->numParameters() : 0);
	if(params.trainNonlinearity) {
		VectorLBFGS nonlParams(y + offset, trainableNonlinearity->numParameters());
		trainableNonlinearity->setParameters(nonlParams);
		offset += trainableNonlinearity->numParameters();
	}

	// initialize gradient and log-likelihood
	if(g) {
		if(params.trainWeights)
			weightsGrad.setZero();
		if(params.trainBias)
			*biasGrad = 0.;
		if(params.trainNonlinearity)
			nonlinearityGrad.setZero();
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
		Array<double, 1, Dynamic> means = mNonlinearity->operator()(responses);

		if(g) {
			Array<double, 1, Dynamic> tmp1 = mDistribution->gradient(output, means);
			
			if(params.trainWeights || params.trainBias) {
				Array<double, 1, Dynamic> tmp2 = differentiableNonlinearity->derivative(responses);
				Array<double, 1, Dynamic> tmp3 = tmp1 * tmp2;

				// weights gradient
				if(params.trainWeights && mDimIn) {
					VectorXd weightsGrad_ = (input.array().rowwise() * tmp3).rowwise().sum();

					#pragma omp critical
					weightsGrad += weightsGrad_;
				}

				// bias gradient
				if(params.trainBias)
					#pragma omp critical
					*biasGrad += tmp3.sum();
			}

			if(params.trainNonlinearity) {
				VectorXd nonlinearityGrad_ = (trainableNonlinearity->gradient(responses).rowwise() * tmp1).rowwise().sum();

				#pragma omp critical
				nonlinearityGrad += nonlinearityGrad_;
			}
		}

		#pragma omp critical
		logLik += mDistribution->logLikelihood(output, means).sum();
	}

	double normConst = outputCompl.cols() * log(2.);

	if(g) {
		for(int i = 0; i < offset; ++i)
			g[i] /= normConst;

		switch(params.regularizer) {
			case Parameters::L1:
				if(params.trainWeights && params.regularizeWeights > 0.)
					weightsGrad += params.regularizeWeights * signum(weights);

				if(params.trainBias && params.regularizeBias > 0.)
					if(bias > 0.)
						*biasGrad += params.regularizeBias;
					else if(bias < 0.)
						*biasGrad -= params.regularizeBias;

				break;

			case Parameters::L2:
				if(params.trainWeights && params.regularizeWeights > 0.)
					weightsGrad += params.regularizeWeights * 2. * weights;

				if(params.trainBias && params.regularizeBias > 0.)
					*biasGrad += params.regularizeBias * 2. * bias;

				break;
		}
	}

	double value = -logLik / normConst;

	switch(params.regularizer) {
		case Parameters::L1:
			if(params.trainWeights && params.regularizeWeights > 0.)
				value += params.regularizeWeights * weights.array().abs().sum();

			if(params.trainBias && params.regularizeBias > 0.)
				value += params.regularizeBias * abs(bias);

			break;

		case Parameters::L2:
			if(params.trainWeights && params.regularizeWeights > 0.)
				value += params.regularizeWeights * weights.array().square().sum();

			if(params.trainBias && params.regularizeBias > 0.)
				value += params.regularizeBias * bias * bias;
			
			break;
	}
 	
 	return value;
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
