#include "conditionaldistribution.h"

#include <cmath>
using std::log;

#include <limits>
using std::numeric_limits;

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
	valIter = 5;
	valLookAhead = 0;
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
	cbIter(params.cbIter),
	valIter(params.valIter),
	valLookAhead(params.valLookAhead)
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
	valIter = params.valIter;
	valLookAhead = params.valLookAhead;

	return *this;
}



ConditionalDistribution::~ConditionalDistribution() {
}



Array<double, 1, Dynamic> ConditionalDistribution::logLikelihood(
	const pair<ArrayXXd, ArrayXXd>& data) const
{
	return logLikelihood(data.first, data.second);
}



double ConditionalDistribution::evaluate(
	const MatrixXd& input,
	const MatrixXd& output) const
{
	return -logLikelihood(input, output).mean() / log(2.) / dimOut();
}



double ConditionalDistribution::evaluate(
	const pair<ArrayXXd, ArrayXXd>& data) const
{
	return -logLikelihood(data.first, data.second).mean() / log(2.) / dimOut();
}
