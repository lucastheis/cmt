#include <cmath>
using std::log;

#include "exception.h"
#include "glm.h"

CMT::GLM::GLM(
	int dimIn,
	Nonlinearity* nonlinearity,
	UnivariateDistribution* distribution) :
	mDimIn(dimIn),
	mNonlinearity(nonlinearity),
	mDistribution(distribution)
{
	mWeights = VectorXd::Random(dimIn) / 100.;
}



CMT::GLM::GLM(const GLM& glm) :
	mDimIn(glm.mDimIn),
	mNonlinearity(glm.mNonlinearity->copy()),
	mDistribution(glm.mDistribution->copy())
{
}



CMT::GLM& CMT::GLM::operator=(const GLM& glm) {
	mDimIn = glm.mDimIn;
	mNonlinearity = glm.mNonlinearity->copy();
	mDistribution = glm.mDistribution->copy();
}



CMT::GLM::~GLM() {
	delete mNonlinearity;
	delete mDistribution;
}



Array<double, 1, Dynamic> CMT::GLM::logLikelihood(
	const MatrixXd& input,
	const MatrixXd& output) const
{
	return mDistribution->logLikelihood(
		output,
		(*mNonlinearity)(mWeights.transpose() * input));
}



int CMT::GLM::numParameters(const Parameters& params) const {
	return mWeights.size();
}



lbfgsfloatval_t* CMT::GLM::parameters(const Parameters& params) const {
	lbfgsfloatval_t* x = lbfgs_malloc(mWeights.size());

	for(int i = 0; i < mWeights.size(); ++i)
		x[i] = mWeights.data()[i];

	return x;
}



void CMT::GLM::setParameters(const lbfgsfloatval_t* x, const Parameters& params) {
	for(int i = 0; i < mWeights.size(); ++i)
		mWeights.data()[i] = x[i];
}



double CMT::GLM::parameterGradient(
	const MatrixXd& input,
	const MatrixXd& output,
	const lbfgsfloatval_t* x,
	lbfgsfloatval_t* g,
	const Parameters& params) const
{
	VectorLBFGS weights(const_cast<lbfgsfloatval_t*>(x), mWeights.size());
	VectorLBFGS weightsGrad(g, mWeights.size());

	Array<double, 1, Dynamic> responses = weights.transpose() * input;
	Array<double, 1, Dynamic> means = (*mNonlinearity)(responses);

	weightsGrad = (
		mDistribution->gradient(output, means) *
		mNonlinearity->derivative(responses) *
		output.array()).colwise().mean() / log(2.);

	return mDistribution->logLikelihood(output, means).mean() / log(2.);
}



CMT::LogisticFunction* CMT::LogisticFunction::copy() {
	return new LogisticFunction(*this);
}



Array<double, 1, Dynamic> CMT::LogisticFunction::operator()(const Array<double, 1, Dynamic>& data) const {
	return 1. / (1. + (-data).exp());
}



Array<double, 1, Dynamic> CMT::LogisticFunction::derivative(const Array<double, 1, Dynamic>& data) const {
	Array<double, 1, Dynamic> tmp = operator()(data);
	return tmp * (1. - tmp);
}



CMT::Bernoulli* CMT::Bernoulli::copy() {
	return new Bernoulli(*this);
}



CMT::Bernoulli::Bernoulli(double prob) : mProb(prob) {
	if(prob < 0. || prob > 1.)
		throw Exception("Probability has to be between 0 and 1.");
}



MatrixXd CMT::Bernoulli::sample(int numSamples) const {
	return (Array<double, 1, Dynamic>::Random(numSamples).abs() > mProb).cast<double>();
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
		grad[i] = data[i] > 0.5 ? -1. / means[i] : 1. / (means[i] - 1.);

	return grad;
}
