#include "univariatedistributions.h"
#include "utils.h"

#include "Eigen/Core"
using Eigen::MatrixXd;
using Eigen::Array;
using Eigen::Dynamic;

CMT::Bernoulli::Bernoulli(double prob) : mProb(prob) {
	if(prob < 0. || prob > 1.)
		throw Exception("Probability has to be between 0 and 1.");
}



double CMT::Bernoulli::mean() const {
	return probability();
}



void CMT::Bernoulli::setMean(double mean) {
	setProbability(mean);
}



MatrixXd CMT::Bernoulli::sample(int numSamples) const {
	return (Array<double, 1, Dynamic>::Random(numSamples).abs() < mProb).cast<double>();
}



MatrixXd CMT::Bernoulli::sample(const Array<double, 1, Dynamic>& means) const {
	return (Array<double, 1, Dynamic>::Random(means.size()).abs() < means).cast<double>();
}



Array<double, 1, Dynamic> CMT::Bernoulli::logLikelihood(const MatrixXd& data) const {
	if(mProb > 0. && 1. - mProb > 0.)
		return log(mProb) * data.array() + log(1. - mProb) * (1. - data.array());

	Array<double, 1, Dynamic> logLik = Array<double, 1, Dynamic>(data.size());

	double logProb1 = log(mProb);
	double logProb0 = log(1. - mProb);

	for(int i = 0; i < data.size(); ++i)
		logLik[i] = data.data()[i] > 0.5 ? logProb1 : logProb0;

	return logLik;
}



Array<double, 1, Dynamic> CMT::Bernoulli::logLikelihood(
	const Array<double, 1, Dynamic>& data,
	const Array<double, 1, Dynamic>& means) const
{
	Array<double, 1, Dynamic> logLik = Array<double, 1, Dynamic>(data.size());

	for(int i = 0; i < data.size(); ++i)
		logLik[i] = data[i] > 0.5 ? log(means[i]) : log(1. - means[i]);

	return logLik;
}



Array<double, 1, Dynamic> CMT::Bernoulli::gradient(
	const Array<double, 1, Dynamic>& data,
	const Array<double, 1, Dynamic>& means) const
{
	Array<double, 1, Dynamic> grad = Array<double, 1, Dynamic>(data.size());

	for(int i = 0; i < data.size(); ++i)
		grad[i] = data[i] > 0.5 ? -1. / means[i] : 1. / (1. - means[i]);

	return grad;
}



CMT::Poisson::Poisson(double lambda) : mLambda(lambda) {
	if(lambda < 0.)
		throw Exception("Lambda has to be non-negative.");
}



double CMT::Poisson::mean() const {
	return mLambda;
}



void CMT::Poisson::setMean(double mean) {
	if(mean < 0.)
		throw Exception("The mean cannot be negative.");
	mLambda = mean;
}



MatrixXd CMT::Poisson::sample(int numSamples) const {
	return samplePoisson(1, numSamples, mLambda).cast<double>();
}



MatrixXd CMT::Poisson::sample(const Array<double, 1, Dynamic>& means) const {
	return samplePoisson(means).cast<double>();
}



Array<double, 1, Dynamic> CMT::Poisson::logLikelihood(const MatrixXd& data) const {
	return data.array() * log(mLambda) - lnGamma(data.array() + 1.) - mLambda;
}



Array<double, 1, Dynamic> CMT::Poisson::logLikelihood(
	const Array<double, 1, Dynamic>& data,
	const Array<double, 1, Dynamic>& means) const
{
	return data * log(means) - lnGamma(data + 1.) - means;
}



Array<double, 1, Dynamic> CMT::Poisson::gradient(
	const Array<double, 1, Dynamic>& data,
	const Array<double, 1, Dynamic>& means) const
{
	return 1. - data / means;
}
