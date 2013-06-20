#include "mixture.h"
#include "utils.h"

#include <cmath>
using std::log;

#include "Eigen/Core"
using Eigen::Dynamic;
using Eigen::Array;
using Eigen::ArrayXd;
using Eigen::ArrayXXd;
using Eigen::MatrixXd;

#include <iostream>

CMT::Mixture::Parameters::Parameters() :
	verbosity(1),
	maxIter(20),
	valIter(2),
	valLookAhead(5),
	trainPriors(true),
	trainComponents(true),
	regularizePriors(0.)
{
}



CMT::Mixture::Component::Parameters::Parameters() :
	verbosity(0),
	maxIter(10),
	trainPriors(true),
	trainCovariance(true),
	trainScales(true),
	trainMean(true),
	regularizePriors(0.),
	regularizeCovariance(0.),
	regularizeScales(0.),
	regularizeMean(0.)
{
}



void CMT::Mixture::Component::initialize(
	const MatrixXd& data,
	const Parameters& parameters) 
{
}



CMT::Mixture::Mixture(int dim) : mDim(dim) {
}



CMT::Mixture::~Mixture() {
	for(int i = 0; i < mComponents.size(); ++i)
		delete mComponents[i];
}



void CMT::Mixture::addComponent(Component* component) {
	if(component->dim() != dim())
		throw Exception("Component has wrong dimensionality.");

	// add another parameter to prior weights vector
	VectorXd priors(numComponents() + 1);
	priors << mPriors * numComponents(), 1.;

	// renormalize
	mPriors = priors / (numComponents() + 1.);

	mComponents.push_back(component);
}



MatrixXd CMT::Mixture::sample(int numSamples) const {
	// cumulative distribution function
	ArrayXd cdf = mPriors;
	for(int k = 1; k < numComponents(); ++k)
		cdf[k] += cdf[k - 1];

	// make sure last entry is definitely large enough
	cdf[numComponents() - 1] = 1.0001;

	// initialize sample from multinomial distribution
	int numSamplesPerComp[numComponents()];
	for(int k = 0; k < numComponents(); ++k)
		numSamplesPerComp[k] = 0;

	// generate sample from multinomial distribution
	#pragma omp parallel for
	for(int i = 0; i < numSamples; ++i) {
		double urand = static_cast<double>(rand()) / (static_cast<long>(RAND_MAX) + 1l);

		int j = 0;
		while(urand > cdf[j])
			++j;

		numSamplesPerComp[j]++;
	}

	// container for samples from different components
	vector<MatrixXd> samples(numComponents());

	// sample each component
	#pragma omp parallel for
	for(int k = 0; k < numComponents(); ++k)
		samples[k] = mComponents[k]->sample(numSamplesPerComp[k]);

	return concatenate(samples);
}



ArrayXXd CMT::Mixture::posterior(const MatrixXd& data) {
	ArrayXXd logJoint(numComponents(), data.cols());

	#pragma omp parallel for
	for(int k = 0; k < numComponents(); ++k)
		logJoint.row(k) = mComponents[k]->logLikelihood(data) + log(mPriors[k]);

	// return normalized posterior
	return (logJoint.rowwise() - logSumExp(logJoint)).exp();
}



Array<double, 1, Dynamic> CMT::Mixture::logLikelihood(const MatrixXd& data) const {
	ArrayXXd logJoint(numComponents(), data.cols());

	#pragma omp parallel for
	for(int k = 0; k < numComponents(); ++k)
		logJoint.row(k) = mComponents[k]->logLikelihood(data) + log(mPriors[k]);

	return logSumExp(logJoint);
}



void CMT::Mixture::initialize(
	const MatrixXd& data,
	const Parameters& parameters,
	const Component::Parameters& componentParameters)
{
	if(parameters.trainPriors)
		mPriors.setConstant(1. / numComponents()); 

	#pragma omp parallel for
	for(int k = 0; k < numComponents(); ++k)
		mComponents[k]->initialize(data, componentParameters);

	mInitialized = true;
}



bool CMT::Mixture::train(
	const MatrixXd& data,
	const Parameters& parameters,
	const Component::Parameters& componentParameters)
{
	if(!initialized())
		initialize(data, parameters, componentParameters);

	ArrayXd postSum;
	ArrayXXd post;
	ArrayXXd weights;

	for(int i = 0; i < parameters.maxIter; ++i) {
		// compute responsibilities (E)
		post = posterior(data);
		postSum = post.rowwise().sum();
		weights = post.colwise() / postSum;

		// optimize prior weights (M)
		if(parameters.trainPriors) {
			mPriors = postSum / data.cols() + parameters.regularizePriors;
			mPriors /= mPriors.sum();
		}

		// optimize components (M)
		if(parameters.trainComponents) {
			#pragma omp parallel for
			for(int k = 0; k < numComponents(); ++k)
				mComponents[k]->train(data, weights.row(k), componentParameters);
		} else {
			break;
		}
	}

	return true;
}
