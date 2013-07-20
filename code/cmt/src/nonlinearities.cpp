#include "nonlinearities.h"

#include "Eigen/Core"
using Eigen::ArrayXXd;

CMT::Nonlinearity::~Nonlinearity() {
}



ArrayXXd CMT::LogisticFunction::operator()(const ArrayXXd& data) const {
	return 1. / (1. + (-data).exp());
}



ArrayXXd CMT::LogisticFunction::derivative(const ArrayXXd& data) const {
	ArrayXXd tmp = operator()(data);
	return tmp * (1. - tmp);
}
