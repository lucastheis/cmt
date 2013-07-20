#include "nonlinearities.h"

#include <cmath>
using std::exp;
using std::log;

#include "Eigen/Core"
using Eigen::ArrayXXd;

CMT::Nonlinearity::~Nonlinearity() {
}



ArrayXXd CMT::LogisticFunction::operator()(const ArrayXXd& data) const {
	return 1. / (1. + (-data).exp());
}



double CMT::LogisticFunction::operator()(double data) const {
	return 1. / (1. + exp(-data));
}



ArrayXXd CMT::LogisticFunction::derivative(const ArrayXXd& data) const {
	ArrayXXd tmp = operator()(data);
	return tmp * (1. - tmp);
}



ArrayXXd CMT::LogisticFunction::inverse(const ArrayXXd& data) const {
	return (data / (1. - data)).log();
}



double CMT::LogisticFunction::inverse(double data) const {
	return log(data / (1. - data));
}
