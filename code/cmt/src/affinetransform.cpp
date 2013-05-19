#include "utils.h"
#include "exception.h"
#include "affinetransform.h"
#include "Eigen/LU"

#include "Eigen/Core"
using Eigen::ArrayXXd;

#include <utility>
using std::pair;


CMT::AffineTransform::AffineTransform(const VectorXd& meanIn, const MatrixXd& preIn, int dimOut) :
	AffinePreconditioner(
		meanIn,
		VectorXd::Zero(dimOut),
		preIn,
		preIn.inverse(),
		MatrixXd::Identity(dimOut, dimOut),
		MatrixXd::Identity(dimOut, dimOut),
		MatrixXd::Zero(dimOut, preIn.rows()))
{
	mGradTransform = MatrixXd::Zero(dimOut, meanIn.size());
	mLogJacobian = 0.;
}



CMT::AffineTransform::AffineTransform(
	const VectorXd& meanIn,
	const MatrixXd& preIn,
	const MatrixXd& preInInv,
	int dimOut) :
	AffinePreconditioner(
		meanIn,
		VectorXd::Zero(dimOut),
		preIn,
		preInInv,
		MatrixXd::Identity(dimOut, dimOut),
		MatrixXd::Identity(dimOut, dimOut),
		MatrixXd::Zero(dimOut, preIn.rows()))
{
	mGradTransform = MatrixXd::Zero(dimOut, meanIn.size());
	mLogJacobian = 0.;
}



CMT::AffineTransform::AffineTransform() {
}



CMT::AffineTransform::AffineTransform(const AffineTransform& transform) :
	AffinePreconditioner(transform)
{
}



pair<ArrayXXd, ArrayXXd> CMT::AffineTransform::operator()(
	const ArrayXXd& input,
	const ArrayXXd& output) const
{
	return make_pair(AffinePreconditioner::operator()(input), output);
}



pair<ArrayXXd, ArrayXXd> CMT::AffineTransform::inverse(
	const ArrayXXd& input,
	const ArrayXXd& output) const
{
	return make_pair(AffinePreconditioner::inverse(input), output);
}



pair<ArrayXXd, ArrayXXd> CMT::AffineTransform::adjustGradient(
	const ArrayXXd& inputGradient,
	const ArrayXXd& outputGradient) const
{
	return make_pair(mPreIn.transpose() * inputGradient.matrix(), outputGradient);
}
