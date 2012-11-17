#include "utils.h"
#include "whiteningtransform.h"

#include "Eigen/Eigenvalues"
using Eigen::SelfAdjointEigenSolver;

MCM::WhiteningTransform::WhiteningTransform(const ArrayXXd& data) : 
	MCM::LinearTransform(SelfAdjointEigenSolver<MatrixXd>(covariance(data)).operatorInverseSqrt())
{
}
