#include "mixture.h"
#include "utils.h"

#include <cmath>
using std::log;

#include <limits>
using std::numeric_limits;

#include <iostream>
using std::cout;
using std::endl;

#include <iomanip>
using std::setw;
using std::setprecision;

#include "Eigen/Core"
using Eigen::Dynamic;
using Eigen::Array;
using Eigen::ArrayXd;
using Eigen::ArrayXXd;
using Eigen::MatrixXd;

CMT::Mixture::Parameters::Parameters() :
	verbosity(1),
	maxIter(20),
	threshold(1e-8),
	valIter(2),
	valLookAhead(5),
	initialize(true),
	trainPriors(true),
	trainComponents(true),
	regularizePriors(0.)
{
}



CMT::Mixture::Component::Parameters::Parameters() :
	verbosity(0),
	maxIter(10),
	threshold(1e-8),
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



CMT::Mixture::Mixture(int dim) : mDim(dim), mInitialized(false) {
}



CMT::Mixture::~Mixture() {
	for(int k = 0; k < mComponents.size(); ++k)
		delete mComponents[k];
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
		double urand = static_cast<double>(rand()) / RAND_MAX;

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
	if(data.rows() != dim())
		throw Exception("Data has wrong dimensionality.");

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
	if(data.rows() != dim())
		throw Exception("Data has wrong dimensionality.");

	if(parameters.initialize && !initialized())
		initialize(data, parameters, componentParameters);

	ArrayXXd logJoint(numComponents(), data.cols());
	Array<double, Dynamic, 1> postSum;
	Array<double, 1, Dynamic> logLik;
	ArrayXXd post;
	ArrayXXd weights;
	double avgLogLoss = numeric_limits<double>::infinity();
	double avgLogLossNew;

	for(int i = 0; i < parameters.maxIter; ++i) {
		// compute joint probability of data and assignments (E)
		#pragma omp parallel for
		for(int k = 0; k < numComponents(); ++k)
			logJoint.row(k) = mComponents[k]->logLikelihood(data) + log(mPriors[k]);

		// compute normalized posterior (E)
		logLik = logSumExp(logJoint);

		// average negative log-likelihood in bits per component
		avgLogLossNew = -logLik.mean() / log(2.) / dim();

		if(parameters.verbosity > 0)
			cout << setw(6) << i << setw(14) << setprecision(7) << avgLogLossNew << endl;

		// test for convergence
		if(avgLogLoss - avgLogLossNew < parameters.threshold)
			return true;
		avgLogLoss = avgLogLossNew;

		// compute normalized posterior (E)
		post = (logJoint.rowwise() - logLik).exp();
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
			return true;
		}
	}

	if(parameters.verbosity > 0)
		cout << setw(6) << parameters.maxIter << setw(14) << setprecision(7) << evaluate(data) << endl;

	return false;
}



bool CMT::Mixture::train(
	const MatrixXd& data,
	const MatrixXd& dataValid,
	const Parameters& parameters,
	const Component::Parameters& componentParameters)
{
	if(parameters.initialize && !initialized())
		initialize(data, parameters, componentParameters);

	ArrayXXd logJoint(numComponents(), data.cols());
	Array<double, Dynamic, 1> postSum;
	Array<double, 1, Dynamic> logLik;
	ArrayXXd post;
	ArrayXXd weights;

	// training and validation log-loss for checking convergence
	double avgLogLoss = numeric_limits<double>::infinity();
	double avgLogLossNew;
	double avgLogLossValid = evaluate(dataValid);
	double avgLogLossValidNew = avgLogLossValid;
	int counter = 0;

	// backup model parameters
	VectorXd priors = mPriors;
	vector<Component*> components;

	for(int k = 0; k < numComponents(); ++k)
		components.push_back(mComponents[k]->copy());

	for(int i = 0; i < parameters.maxIter; ++i) {
		// compute joint probability of data and assignments (E)
		#pragma omp parallel for
		for(int k = 0; k < numComponents(); ++k)
			logJoint.row(k) = mComponents[k]->logLikelihood(data) + log(mPriors[k]);

		// compute normalized posterior (E)
		logLik = logSumExp(logJoint);

		// average negative log-likelihood in bits per component
		avgLogLossNew = -logLik.mean() / log(2.) / dim();

		if(parameters.verbosity > 0) {
			if(i % parameters.valIter == 0) {
				// print training and validation error
				cout << setw(6) << i;
				cout << setw(14) << setprecision(7) << avgLogLossNew;
				cout << setw(14) << setprecision(7) << avgLogLossValidNew << endl;
			} else {
				// print training error
				cout << setw(6) << i << setw(14) << setprecision(7) << avgLogLossNew << endl;
			}
		}

		// test for convergence
		if(avgLogLoss - avgLogLossNew < parameters.threshold)
			return true;
		avgLogLoss = avgLogLossNew;

		// compute normalized posterior (E)
		post = (logJoint.rowwise() - logLik).exp();
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
			return true;
		}

		if((i + 1) % parameters.valIter == 0) {
			// check validation error
			avgLogLossValidNew = evaluate(dataValid);

			if(avgLogLossValidNew < avgLogLossValid) {
				// backup new found model parameters
				priors = mPriors;
				for(int k = 0; k < numComponents(); ++k)
					*components[k] = *mComponents[k];
				
				avgLogLossValid = avgLogLossValidNew;
			} else {
				counter++;

				if(parameters.valLookAhead > 0 && counter >= parameters.valLookAhead) {
					// set parameters to best parameters found during training
					mPriors = priors;

					for(int k = 0; k < numComponents(); ++k) {
						*mComponents[k] = *components[k];
						delete components[k];
					}

					return true;
				}
			}
		}
	}

	if(parameters.verbosity > 0)
		cout << setw(6) << parameters.maxIter << setw(11) << setprecision(5) << evaluate(data) << endl;

	return false;
}
