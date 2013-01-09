#include "exception.h"
#include "utils.h"
#include "lineartransform.h"

#include <iostream>

MCM::AffineTransform::AffineTransform(const MatrixXd& mat, const VectorXd& vec) :
	mMat(mat),
	mVec(vec)
{
}



int MCM::AffineTransform::dimIn() const {
	return mMat.cols();
}



int MCM::AffineTransform::dimOut() const {
	return mMat.rows();
}



ArrayXXd MCM::AffineTransform::operator()(const ArrayXXd& data) const {
	if(data.rows() != mMat.cols())
		throw Exception("Data has wrong dimensionality.");
	if(mMat.rows() != mVec.rows())
		throw Exception("Matrix and vector of affine transformation are incompatible.");
	return (mMat * data.matrix()).colwise() + mVec;
}



ArrayXXd MCM::AffineTransform::inverse(const ArrayXXd& data) const {
	if(data.rows() != mMat.rows())
		throw Exception("Data has wrong dimensionality.");
	if(mMat.rows() != mVec.rows())
		throw Exception("Matrix and vector of affine transformation are incompatible.");
	if(!mMatInverse.size())
		mMatInverse = pInverse(mMat);
	return mMatInverse * (data.matrix().colwise() - mVec);
}
