#include "exception.h"
#include "utils.h"
#include "lineartransform.h"

MCM::LinearTransform::LinearTransform(const MatrixXd& mat) : 
	AffineTransform(mat, VectorXd::Zero(mat.rows()))
{
}



ArrayXXd MCM::LinearTransform::operator()(const ArrayXXd& data) const {
	if(data.rows() != mMat.cols())
		throw Exception("Data has wrong dimensionality.");
	return mMat * data.matrix();
}



ArrayXXd MCM::LinearTransform::inverse(const ArrayXXd& data) const {
	if(data.rows() != mMat.rows())
		throw Exception("Data has wrong dimensionality.");
	if(!mMatInverse.size())
		mMatInverse = pInverse(mMat);
	return mMatInverse * data.matrix();
}
