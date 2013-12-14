#include "Eigen/Core"
using Eigen::Dynamic;
using Eigen::Array;
using Eigen::MatrixXd;

#include "conditionaldistribution.h"
#include "exception.h"

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
	return -logLikelihood(preconditioner(input, output)).mean() / log(2.) / dimOut()
		- preconditioner.logJacobian(input, output).mean() / log(2.) / dimOut();
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
	return -logLikelihood(preconditioner(data.first, data.second)).mean() / log(2.) / dimOut()
		- preconditioner.logJacobian(data).mean() / log(2.) / dimOut();
}



/**
 * Computes the expectation value of the output.
 */
MatrixXd CMT::ConditionalDistribution::predict(const MatrixXd& input) const {
	throw Exception("This method is not yet implemented.");
}
