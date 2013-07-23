#include "binningtransform.h"
#include "exception.h"
#include "utils.h"

CMT::BinningTransform::BinningTransform(int binning, int dimIn, int dimOut) :
	AffineTransform(
		VectorXd::Zero(dimIn),
		MatrixXd::Zero(dimIn / binning, dimIn),
		dimOut),
	mBinning(binning)
{
	if(dimIn % binning)
		throw Exception("Input dimensionality has to be a multiple of binning width.");

	for(int i = 0, j = 0; i < mPreIn.rows(); ++i)
		for(int k = 0; k < binning; ++k, ++j)
			mPreIn(i, j) = 1.;

	mPreInInv = pInverse(mPreIn);
}
