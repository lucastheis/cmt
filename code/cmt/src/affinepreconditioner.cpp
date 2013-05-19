#include "utils.h"
#include "exception.h"
#include "affinepreconditioner.h"
#include "Eigen/LU"

#include "Eigen/Core"
using Eigen::Dynamic;
using Eigen::Array;
using Eigen::ArrayXXd;

#include <utility>
using std::pair;
using std::make_pair;

CMT::AffinePreconditioner::AffinePreconditioner(
	const VectorXd& meanIn,
	const VectorXd& meanOut,
	const MatrixXd& preIn,
	const MatrixXd& preOut,
	const MatrixXd& predictor) :
	mMeanIn(meanIn),
	mMeanOut(meanOut),
	mPreIn(preIn),
	mPreInInv(preIn.inverse()),
	mPreOut(preOut),
	mPreOutInv(preOut.inverse()),
	mPredictor(predictor),
	mLogJacobian(preOut.partialPivLu().matrixLU().diagonal().array().abs().log().sum())
{
	if(preIn.size() > 0)
		mGradTransform = preOut * predictor * preIn;
	else
		mGradTransform = MatrixXd(preOut.rows(), preIn.cols());
}



CMT::AffinePreconditioner::AffinePreconditioner(
	const VectorXd& meanIn,
	const VectorXd& meanOut,
	const MatrixXd& preIn,
	const MatrixXd& preInInv,
	const MatrixXd& preOut,
	const MatrixXd& preOutInv,
	const MatrixXd& predictor) :
	mMeanIn(meanIn),
	mMeanOut(meanOut),
	mPreIn(preIn),
	mPreInInv(preInInv),
	mPreOut(preOut),
	mPreOutInv(preOutInv),
	mPredictor(predictor),
	mLogJacobian(preOut.partialPivLu().matrixLU().diagonal().array().abs().log().sum())
{
	if(preIn.size() > 0)
		mGradTransform = preOut * predictor * preIn;
	else
		mGradTransform = MatrixXd(preOut.rows(), preIn.cols());
}



CMT::AffinePreconditioner::AffinePreconditioner(const AffinePreconditioner& preconditioner) :
	mMeanIn(preconditioner.mMeanIn),
	mMeanOut(preconditioner.mMeanOut),
	mPreIn(preconditioner.mPreIn),
	mPreInInv(preconditioner.mPreInInv),
	mPreOut(preconditioner.mPreOut),
	mPreOutInv(preconditioner.mPreOutInv),
	mPredictor(preconditioner.mPredictor),
	mLogJacobian(preconditioner.mLogJacobian),
	mGradTransform(preconditioner.mGradTransform)
{
}



CMT::AffinePreconditioner::AffinePreconditioner() {
}



CMT::AffinePreconditioner::~AffinePreconditioner() {
}



int CMT::AffinePreconditioner::dimIn() const {
	return mMeanIn.size();
}



int CMT::AffinePreconditioner::dimInPre() const {
	return mPreIn.rows();
}



int CMT::AffinePreconditioner::dimOut() const {
	return mMeanOut.size();
}



int CMT::AffinePreconditioner::dimOutPre() const {
	return mPreOut.rows();
}



pair<ArrayXXd, ArrayXXd> CMT::AffinePreconditioner::operator()(
	const ArrayXXd& input,
	const ArrayXXd& output) const
{
	if(input.cols() != output.cols())
		throw Exception("Number of inputs and outputs must be the same."); 
	if(input.rows() != dimIn())
		throw Exception("Input has wrong dimensionality."); 
	if(output.rows() != dimOut())
		throw Exception("Output has wrong dimensionality."); 

	if(input.rows() < 1) {
		ArrayXXd outputTr = mPreOut * (output.matrix().colwise() - mMeanOut);
		return make_pair(input, outputTr);
	} else {
		ArrayXXd inputTr = mPreIn * (input.matrix().colwise() - mMeanIn);
		ArrayXXd outputTr = mPreOut * (output.matrix().colwise() - mMeanOut - mPredictor * inputTr.matrix());
		return make_pair(inputTr, outputTr);
	}
}



pair<ArrayXXd, ArrayXXd> CMT::AffinePreconditioner::inverse(
	const ArrayXXd& input,
	const ArrayXXd& output) const
{
	if(input.cols() != output.cols())
		throw Exception("Number of inputs and outputs must be the same."); 
	if(input.rows() != dimInPre())
		throw Exception("Input has wrong dimensionality."); 
	if(output.rows() != dimOutPre())
		throw Exception("Output has wrong dimensionality."); 
	if(input.rows() < 1) {
		ArrayXXd outputTr = (mPreOutInv * output.matrix()).colwise() + mMeanOut;
		return make_pair(input, outputTr);
	} else {
		ArrayXXd outputTr = (mPreOutInv * output.matrix() + mPredictor * input.matrix()).colwise() + mMeanOut;
		ArrayXXd inputTr = (mPreInInv * input.matrix()).colwise() + mMeanIn;
		return make_pair(inputTr, outputTr);
	}
}



ArrayXXd CMT::AffinePreconditioner::operator()(const ArrayXXd& input) const {
	if(input.rows() != dimIn())
		throw Exception("Input has wrong dimensionality."); 
	if(input.rows() < 1)
		return input;
	return mPreIn * (input.matrix().colwise() - mMeanIn);
}



ArrayXXd CMT::AffinePreconditioner::inverse(const ArrayXXd& input) const {
	if(input.rows() != dimInPre())
		throw Exception("Input has wrong dimensionality."); 
	if(input.rows() < 1)
		return input;
	return (mPreInInv * input.matrix()).colwise() + mMeanIn;
}



Array<double, 1, Dynamic> CMT::AffinePreconditioner::logJacobian(const ArrayXXd& input, const ArrayXXd& output) const {
	return Array<double, 1, Dynamic>::Zero(output.cols()) + mLogJacobian;
}



pair<ArrayXXd, ArrayXXd> CMT::AffinePreconditioner::adjustGradient(
	const ArrayXXd& inputGradient,
	const ArrayXXd& outputGradient) const
{
	if(inputGradient.rows() < 1)
		return make_pair(
			inputGradient,
			mPreOut.transpose() * outputGradient.matrix());
	return make_pair(
		mPreIn.transpose() * inputGradient.matrix() - mGradTransform.transpose() * outputGradient.matrix(),
		mPreOut.transpose() * outputGradient.matrix());
}
