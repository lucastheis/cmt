#include "distribution.h"
#include <cmath>

using std::log;

Distribution::~Distribution() {
}



double Distribution::evaluate(const MatrixXd& data) const {
	return -logLikelihood(data).mean() / log(2.) / dim();
}
