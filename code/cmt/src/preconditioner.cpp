#include "preconditioner.h"

#include "Eigen/Core"
using Eigen::Array;
using Eigen::ArrayXXd;
using Eigen::Dynamic;

#include <utility>
using std::pair;

CMT::Preconditioner::~Preconditioner() {
}



pair<ArrayXXd, ArrayXXd> CMT::Preconditioner::operator()(
	const pair<ArrayXXd, ArrayXXd>& data) const 
{
	return operator()(data.first, data.second);
}



pair<ArrayXXd, ArrayXXd> CMT::Preconditioner::inverse(
	const pair<ArrayXXd, ArrayXXd>& data) const 
{
	return inverse(data.first, data.second);
}



Array<double, 1, Dynamic> CMT::Preconditioner::logJacobian(
	const pair<ArrayXXd, ArrayXXd>& data) const 
{
	return logJacobian(data.first, data.second);
}



pair<ArrayXXd, ArrayXXd> CMT::Preconditioner::adjustGradient(
	const pair<ArrayXXd, ArrayXXd>& dataGradient) const
{
	return adjustGradient(dataGradient.first, dataGradient.second);
}
