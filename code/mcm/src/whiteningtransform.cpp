#include "Eigen/Eigenvalues"
#include "utils.h"
#include "whiteningtransform.h"

MCM::WhiteningTransform::WhiteningTransform(ArrayXXd data) : 
	MCM::LinearTransform(SelfAdjointEigenSolver<MatrixXd>(covariance(data)).operatorInverseSqrt())
{
}
