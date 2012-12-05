#include "identitytransform.h"

int MCM::IdentityTransform::dimIn() const {
	return -1;
}



int MCM::IdentityTransform::dimOut() const {
	return -1;
}



ArrayXXd MCM::IdentityTransform::operator()(const ArrayXXd& input) const {
	return input;
}



ArrayXXd MCM::IdentityTransform::inverse(const ArrayXXd& output) const {
	return output;
}
