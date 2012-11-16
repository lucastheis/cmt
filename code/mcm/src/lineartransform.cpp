#include "exception.h"
#include "utils.h"
#include "lineartransform.h"

MCM::LinearTransform::LinearTransform(MatrixXd mat) : mMat(mat) {
}



ArrayXXd MCM::LinearTransform::operator()(ArrayXXd input) const {
	if(input.rows() != mMat.cols())
		throw Exception("Data has wrong dimensionality.");
	return mMat * input.matrix();
}



ArrayXXd MCM::LinearTransform::inverse(ArrayXXd input) const {
	if(input.rows() != mMat.rows())
		throw Exception("Data has wrong dimensionality.");
	if(!mMatInverse.size())
		mMatInverse = pInverse(mMat);
	return mMatInverse * input.matrix();
}
