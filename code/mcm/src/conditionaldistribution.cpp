#include "conditionaldistribution.h"
#include <cmath>

using std::log;

double ConditionalDistribution::evaluate(const MatrixXd& input, const MatrixXd& output) {
	return -logLikelihood(input, output).mean() / log(2.) / dimOut();
}