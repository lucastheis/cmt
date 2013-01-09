#include "utils.h"
#include "whiteningtransform.h"

#include "Eigen/Eigenvalues"
using Eigen::SelfAdjointEigenSolver;

MCM::WhiteningTransform::WhiteningTransform(const ArrayXXd& data) : 
	MCM::AffineTransform(
		SelfAdjointEigenSolver<MatrixXd>(covariance(data)).operatorInverseSqrt(),
		-data.rowwise().mean())
{
}
