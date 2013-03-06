#include "preconditioner.h"

#include "identitytransform.h"
using MCM::IdentityTransform;

MCM::Preconditioner::Preconditioner(const Transform& transform) :
	mTransform(transform) 
{
}



MCM::Preconditioner::Preconditioner() : mTransform(IdentityTransform()) {
}



int MCM::Preconditioner::dimIn() const {
	return mTransform.dimIn();
}



int MCM::Preconditioner::dimOut() const {
	return -1;
}



pair<ArrayXXd, ArrayXXd> MCM::Preconditioner::operator()(
	const ArrayXXd& input, 
	const ArrayXXd& output) const 
{
	return make_pair(mTransform(input), output);
}



pair<ArrayXXd, ArrayXXd> MCM::Preconditioner::inverse(
	const ArrayXXd& input,
	const ArrayXXd& output) const 
{
	return make_pair(mTransform.inverse(input), output);
}
