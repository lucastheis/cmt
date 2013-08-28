#include "nonlinearities.h"
#include "exception.h"
#include "utils.h"

#include <cmath>
using std::exp;
using std::log;
using std::tanh;

#include "Eigen/Core"
using Eigen::ArrayXd;
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



CMT::ExponentialFunction::ExponentialFunction(double epsilon) : mEpsilon(epsilon) {
}



ArrayXXd CMT::ExponentialFunction::operator()(const ArrayXXd& data) const {
	return data.exp() + mEpsilon;
}



double CMT::ExponentialFunction::operator()(double data) const {
	return exp(data) + mEpsilon;
}



ArrayXXd CMT::ExponentialFunction::derivative(const ArrayXXd& data) const {
	return data.exp();
}



ArrayXXd CMT::ExponentialFunction::inverse(const ArrayXXd& data) const {
	return (data - mEpsilon).log();
}



double CMT::ExponentialFunction::inverse(double data) const {
	return log(data - mEpsilon);
}



CMT::HistogramNonlinearity::HistogramNonlinearity(
	const ArrayXXd& inputs,
	const ArrayXXd& outputs,
	int numBins,
	double epsilon) : mEpsilon(epsilon)
{
	initialize(inputs, outputs, numBins);
}



CMT::HistogramNonlinearity::HistogramNonlinearity(
	const ArrayXXd& inputs,
	const ArrayXXd& outputs,
	const vector<double>& binEdges,
	double epsilon) : mEpsilon(epsilon)
{
	initialize(inputs, outputs, binEdges);
}



CMT::HistogramNonlinearity::HistogramNonlinearity(
	const vector<double>& binEdges,
	double epsilon) : mEpsilon(epsilon), mBinEdges(binEdges)
{
	mHistogram = vector<double>(mBinEdges.size() - 1);

	for(int k = 0; k < mHistogram.size(); ++k)
		mHistogram[k] = 0.;
}



void CMT::HistogramNonlinearity::initialize(
	const ArrayXXd& inputs,
	const ArrayXXd& outputs,
	int numBins)
{
	double max = inputs.maxCoeff();
	double min = inputs.minCoeff();

	mBinEdges = vector<double>(numBins + 1);

	double binWidth = (max - min) / numBins;

	for(int k = 0; k < mBinEdges.size(); ++k)
		mBinEdges[k] = min + k * binWidth;

	initialize(inputs, outputs);
}



/**
 * @param binEdges a list of bin edges sorted in ascending order
 */
void CMT::HistogramNonlinearity::initialize(
	const ArrayXXd& inputs,
	const ArrayXXd& outputs,
	const vector<double>& binEdges)
{
	mBinEdges = binEdges;
	initialize(inputs, outputs);
}



void CMT::HistogramNonlinearity::initialize(
	const ArrayXXd& inputs,
	const ArrayXXd& outputs)
{
	if(inputs.rows() != outputs.rows() || inputs.cols() != outputs.cols())
		throw Exception("Inputs and outputs have to have same size.");

	mHistogram = vector<double>(mBinEdges.size() - 1);
	vector<int> counter(mBinEdges.size() - 1);

	for(int k = 0; k < mHistogram.size(); ++k) {
		mHistogram[k] = 0.;
		counter[k] = 0;
	}

	for(int i = 0; i < inputs.rows(); ++i)
		for(int j = 0; j < inputs.cols(); ++j) {
			// find bin
			int k = bin(inputs(i, j));

			// update histogram
			counter[k] += 1;
			mHistogram[k] += outputs(i, j);
		}

	for(int k = 0; k < mHistogram.size(); ++k)
		if(mHistogram[k] > 0.)
			// average output observed in bin k
			mHistogram[k] /= counter[k];
}



ArrayXXd CMT::HistogramNonlinearity::operator()(const ArrayXXd& inputs) const {
	ArrayXXd outputs(inputs.rows(), inputs.cols());

	for(int i = 0; i < inputs.rows(); ++i)
		for(int j = 0; j < inputs.cols(); ++j)
			outputs(i, j) = mHistogram[bin(inputs(i, j))] + mEpsilon;

	return outputs;
}



double CMT::HistogramNonlinearity::operator()(double input) const {
	return mHistogram[bin(input)] + mEpsilon;
}



/**
 * Finds index into histogram.
 */
int CMT::HistogramNonlinearity::bin(double input) const {
	// find bin
	for(int k = 0; k < mBinEdges.size() - 1; ++k)
		if(input < mBinEdges[k + 1])
			return k;
	return mHistogram.size() - 1;
}




ArrayXd CMT::HistogramNonlinearity::parameters() const {
	ArrayXd histogram(mHistogram.size());

	for(int i = 0; i < mHistogram.size(); ++i)
		histogram[i] = mHistogram[i];

	return histogram;
}



void CMT::HistogramNonlinearity::setParameters(const ArrayXd& parameters) {
	if(parameters.size() != mHistogram.size())
		throw Exception("Wrong number of parameters.");

	for(int i = 0; i < mHistogram.size(); ++i)
		mHistogram[i] = parameters[i];
}



int CMT::HistogramNonlinearity::numParameters() const {
	return mHistogram.size();
}



ArrayXXd CMT::HistogramNonlinearity::gradient(const ArrayXXd& inputs) const {
	if(inputs.rows() != 1)
		throw Exception("Data has to be stored in one row.");

	ArrayXXd gradient = ArrayXXd::Zero(mHistogram.size(), inputs.cols());

	for(int i = 0; i < inputs.rows(); ++i)
		for(int j = 0; j < inputs.rows(); ++j)
			gradient(bin(inputs(i, j)), j) = 1;

	return gradient;
}



CMT::BlobNonlinearity::BlobNonlinearity(int numComponents, double epsilon) :
	mEpsilon(epsilon),
	mNumComponents(numComponents),
	mMeans(sampleNormal(numComponents, 1)),
	mLogPrecisions((ArrayXd::Random(numComponents).abs() + 0.5).log()),
	mLogWeights(((ArrayXd::Random(numComponents).abs() * 0.4 + 0.1) / numComponents).log())
{
}




ArrayXXd CMT::BlobNonlinearity::operator()(const ArrayXXd& inputs) const {
	if(inputs.rows() != 1)
		throw Exception("Data has to be stored in one row.");

	ArrayXXd diff = ArrayXXd::Zero(mNumComponents, inputs.cols());
	diff.rowwise() += inputs.row(0);
	diff.colwise() -= mMeans;

	ArrayXXd negEnergy = diff.square().colwise() * (-mLogPrecisions.exp() / 2.);
	return (mLogWeights.exp().transpose().matrix() * negEnergy.exp().matrix()).array() + mEpsilon;
}



double CMT::BlobNonlinearity::operator()(double input) const {
	ArrayXd negEnergy = -mLogPrecisions.exp() / 2. * (mMeans - input).square();
	return mLogWeights.exp().matrix().dot(negEnergy.exp().matrix()) + mEpsilon;
}



ArrayXd CMT::BlobNonlinearity::parameters() const {
	ArrayXd parameters(3 * mNumComponents);
	parameters << mMeans, mLogPrecisions, mLogWeights;
	return parameters;
}



void CMT::BlobNonlinearity::setParameters(const ArrayXd& parameters) {
	mMeans = parameters.topRows(mNumComponents);
	mLogPrecisions = parameters.middleRows(mNumComponents, mNumComponents);
	mLogWeights = parameters.bottomRows(mNumComponents);
}



int CMT::BlobNonlinearity::numParameters() const {
	return 3 * mNumComponents;
}



ArrayXXd CMT::BlobNonlinearity::gradient(const ArrayXXd& inputs) const {
	if(inputs.rows() != 1)
		throw Exception("Data has to be stored in one row.");
	
	ArrayXXd diff = ArrayXXd::Zero(mNumComponents, inputs.cols());
	diff.rowwise() += inputs.row(0);
	diff.colwise() -= mMeans;

	ArrayXXd diffSq = diff.square();
	ArrayXd precisions = mLogPrecisions.exp();
	ArrayXd weights = mLogWeights.exp();

	ArrayXXd negEnergy = diffSq.colwise() * (-precisions / 2.);
	ArrayXXd negEnergyExp = negEnergy.exp();

	ArrayXXd gradient(3 * mNumComponents, inputs.cols());

	// gradient of mean
	gradient.topRows(mNumComponents) = (diff * negEnergyExp).colwise() * (weights * precisions);

	// gradient of log-precisions
	gradient.middleRows(mNumComponents, mNumComponents) = (diffSq / 2. * negEnergyExp).colwise() * (-weights * precisions);

	// gradient of log-weights
	gradient.bottomRows(mNumComponents) = negEnergyExp.colwise() * weights;

	return gradient;
}



ArrayXXd CMT::BlobNonlinearity::derivative(const ArrayXXd& inputs) const {
	if(inputs.rows() != 1)
		throw Exception("Data has to be stored in one row.");

	ArrayXXd diff = ArrayXXd::Zero(mNumComponents, inputs.cols());
	diff.rowwise() -= inputs.row(0);
	diff.colwise() += mMeans;

	ArrayXd precisions = mLogPrecisions.exp();

	ArrayXXd negEnergy = diff.square().colwise() * (-precisions / 2.);

	return (mLogWeights.exp() * precisions).transpose().matrix() * (diff * negEnergy.exp()).matrix();
}



CMT::TanhBlobNonlinearity::TanhBlobNonlinearity(int numComponents, double epsilon) :
	mNonlinearity(numComponents, epsilon)
{
}




ArrayXXd CMT::TanhBlobNonlinearity::operator()(const ArrayXXd& inputs) const {
	return tanh(mNonlinearity(inputs));
}



double CMT::TanhBlobNonlinearity::operator()(double input) const {
	return std::tanh(mNonlinearity(input));
}



ArrayXd CMT::TanhBlobNonlinearity::parameters() const {
	return mNonlinearity.parameters();
}



void CMT::TanhBlobNonlinearity::setParameters(const ArrayXd& parameters) {
	mNonlinearity.setParameters(parameters);
}



int CMT::TanhBlobNonlinearity::numParameters() const {
	return mNonlinearity.numParameters();
}



ArrayXXd CMT::TanhBlobNonlinearity::gradient(const ArrayXXd& inputs) const {
	return mNonlinearity.gradient(inputs).rowwise() * sech(mNonlinearity(inputs)).square().row(0);
}



ArrayXXd CMT::TanhBlobNonlinearity::derivative(const ArrayXXd& inputs) const {
	return mNonlinearity.derivative(inputs).rowwise() * sech(mNonlinearity(inputs)).square().row(0);
}
