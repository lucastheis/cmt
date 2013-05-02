#include "conditionaldistribution.h"

#include <cmath>
using std::log;

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
