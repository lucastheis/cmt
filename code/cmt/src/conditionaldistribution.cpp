#include "Eigen/Core"
using Eigen::Dynamic;
using Eigen::Array;

#include "conditionaldistribution.h"

#include <cmath>
using std::log;

CMT::ConditionalDistribution::~ConditionalDistribution() {
}



Array<double, 1, Dynamic> CMT::ConditionalDistribution::logLikelihood(
	const pair<ArrayXXd, ArrayXXd>& data) const
{
	return logLikelihood(data.first, data.second);
}



double CMT::ConditionalDistribution::evaluate(
	const MatrixXd& input,
	const MatrixXd& output) const
{
	return -logLikelihood(input, output).mean() / log(2.) / dimOut();
}



double CMT::ConditionalDistribution::evaluate(
	const MatrixXd& input,
	const MatrixXd& output,
	const Preconditioner& preconditioner) const
{
	pair<ArrayXXd, ArrayXXd> dataInv = preconditioner.inverse(input, output);

	return -logLikelihood(input, output).mean() / log(2.) / dimOut()
		- preconditioner.logJacobian(dataInv).mean() / log(2.) / dimOut();
}



double CMT::ConditionalDistribution::evaluate(
	const pair<ArrayXXd, ArrayXXd>& data) const
{
	return -logLikelihood(data.first, data.second).mean() / log(2.) / dimOut();
}



double CMT::ConditionalDistribution::evaluate(
	const pair<ArrayXXd, ArrayXXd>& data,
	const Preconditioner& preconditioner) const
{
	pair<ArrayXXd, ArrayXXd> dataInv = preconditioner.inverse(data.first, data.second);

	return -logLikelihood(data.first, data.second).mean() / log(2.) / dimOut();
		- preconditioner.logJacobian(dataInv).mean() / log(2.) / dimOut();
}
