#include "exception.h"
#include "utils.h"
#include "lineartransform.h"

MCM::LinearTransform::LinearTransform(const MatrixXd& mat) : mMat(mat) {
}



int MCM::LinearTransform::dimIn() const {
	return mMat.cols();
}



int MCM::LinearTransform::dimOut() const {
	return mMat.rows();
}



ArrayXXd MCM::LinearTransform::operator()(const ArrayXXd& input) const {
	if(input.rows() != mMat.cols())
		throw Exception("Data has wrong dimensionality.");
	return mMat * input.matrix();
}



ArrayXXd MCM::LinearTransform::inverse(const ArrayXXd& input) const {
	if(input.rows() != mMat.rows())
		throw Exception("Data has wrong dimensionality.");
	if(!mMatInverse.size())
		mMatInverse = pInverse(mMat);
	return mMatInverse * input.matrix();
}
