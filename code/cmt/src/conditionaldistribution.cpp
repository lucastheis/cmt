#include "conditionaldistribution.h"
#include <cmath>

using std::log;

ConditionalDistribution::Callback::~Callback() {
}



ConditionalDistribution::Parameters::Parameters() {
	verbosity = 0;
	maxIter = 1000;
	threshold = 1e-5;
	numGrad = 20;
	batchSize = 2000;
	callback = 0;
	cbIter = 25;
}



ConditionalDistribution::Parameters::~Parameters() {
	if(callback)
		delete callback;
}



ConditionalDistribution::Parameters::Parameters(const Parameters& params) :
	verbosity(params.verbosity),
	maxIter(params.maxIter),
	threshold(params.threshold),
	numGrad(params.numGrad),
	batchSize(params.batchSize),
	callback(0),
	cbIter(params.cbIter)
{
	if(params.callback)
		callback = params.callback->copy();
}



ConditionalDistribution::Parameters& ConditionalDistribution::Parameters::operator=(
	const Parameters& params)
{
	verbosity = params.verbosity;
	maxIter = params.maxIter;
	threshold = params.threshold;
	numGrad = params.numGrad;
	batchSize = params.batchSize;
	callback = params.callback ? params.callback->copy() : 0;
	cbIter = params.cbIter;

	return *this;
}



ConditionalDistribution::~ConditionalDistribution() {
}



double ConditionalDistribution::evaluate(
	const MatrixXd& input,
	const MatrixXd& output) const
{
	return -logLikelihood(input, output).mean() / log(2.) / dimOut();
}



void ConditionalDistribution::initialize(
	const MatrixXd& input,
	const MatrixXd& output) const
{
}



bool ConditionalDistribution::train(
	const MatrixXd& input,
	const MatrixXd& output) const
{
	return true;
}
