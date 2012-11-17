#include "identitytransform.h"

ArrayXXd MCM::IdentityTransform::operator()(const ArrayXXd& input) const {
	return input;
}



ArrayXXd MCM::IdentityTransform::inverse(const ArrayXXd& output) const {
	return output;
}
