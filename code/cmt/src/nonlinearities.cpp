#include "nonlinearities.h"

#include <cmath>
using std::exp;
using std::log;

#include "Eigen/Core"
using Eigen::ArrayXXd;

CMT::Nonlinearity::~Nonlinearity() {
}



CMT::LogisticFunction::LogisticFunction(double epsilon) : mEpsilon(epsilon) {
}



ArrayXXd CMT::LogisticFunction::operator()(const ArrayXXd& data) const {
	return mEpsilon / 2. + (1. - mEpsilon) / (1. + (-data).exp());
}



double CMT::LogisticFunction::operator()(double data) const {
	return mEpsilon / 2. + (1. - mEpsilon) / (1. + exp(-data));
}



ArrayXXd CMT::LogisticFunction::derivative(const ArrayXXd& data) const {
	ArrayXXd tmp = operator()(data);
	return (1. - mEpsilon) * tmp * (1. - tmp);
}



ArrayXXd CMT::LogisticFunction::inverse(const ArrayXXd& data) const {
	return ((data - mEpsilon / 2.) / (1. - data - mEpsilon / 2.)).log();
}



double CMT::LogisticFunction::inverse(double data) const {
	return log((data - mEpsilon / 2.) / (1. - data - mEpsilon / 2.));
}



CMT::ExponentialFunction::ExponentialFunction() {
}



ArrayXXd CMT::ExponentialFunction::operator()(const ArrayXXd& data) const {
	return data.exp();
}



double CMT::ExponentialFunction::operator()(double data) const {
	return exp(data);
}



ArrayXXd CMT::ExponentialFunction::derivative(const ArrayXXd& data) const {
	return data.exp();
}



ArrayXXd CMT::ExponentialFunction::inverse(const ArrayXXd& data) const {
	return data.log();
}



double CMT::ExponentialFunction::inverse(double data) const {
	return log(data);
}
