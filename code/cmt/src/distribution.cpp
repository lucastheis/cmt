#include "distribution.h"
#include <cmath>

using std::log;

CMT::Distribution::~Distribution() {
}



/**
 * Computes the average negative log-likelihood in bits per component.
 */
double CMT::Distribution::evaluate(const MatrixXd& data) const {
	return -logLikelihood(data).mean() / log(2.) / dim();
}
