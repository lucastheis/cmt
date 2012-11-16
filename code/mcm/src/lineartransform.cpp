#include "utils.h"
#include "lineartransform.h"

MCM::LinearTransform::LinearTransform(MatrixXd mat) : mMat(mat) {
}



ArrayXXd MCM::LinearTransform::operator()(ArrayXXd input) const {
	return mMat * input.matrix();
}



ArrayXXd MCM::LinearTransform::inverse(ArrayXXd input) const {
	if(!mMatInverse.size())
		mMatInverse = pInverse(mMat);
	return mMatInverse * input.matrix();
}
