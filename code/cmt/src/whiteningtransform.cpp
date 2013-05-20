#include "utils.h"
#include "exception.h"
#include "whiteningtransform.h"

#include "Eigen/Eigenvalues"
using Eigen::SelfAdjointEigenSolver;

CMT::WhiteningTransform::WhiteningTransform(const ArrayXXd& input, const ArrayXXd& output) {
	initialize(input, output.rows());
}



CMT::WhiteningTransform::WhiteningTransform(const ArrayXXd& input, int dimOut) {
	initialize(input, dimOut);
}



CMT::WhiteningTransform::WhiteningTransform(
	const VectorXd& meanIn,
	const MatrixXd& preIn,
	const MatrixXd& preInInv,
	int dimOut) : AffineTransform(meanIn, preIn, preInInv, dimOut)
{
}



void CMT::WhiteningTransform::initialize(const ArrayXXd& input, int dimOut) {
	if(input.cols() < input.rows())
		throw Exception("Too few inputs to compute whitening transform."); 

	mMeanIn = input.rowwise().mean();

	// compute covariances
	MatrixXd covXX = covariance(input);

	// input whitening
	SelfAdjointEigenSolver<MatrixXd> eigenSolver;

	eigenSolver.compute(covXX);

	Array<double, 1, Dynamic> eigenvalues = eigenSolver.eigenvalues();
	MatrixXd eigenvectors = eigenSolver.eigenvectors();

	// don't whiten directions with near-zero variance
	for(int i = 0; i < eigenvalues.size(); ++i)
		if(eigenvalues[i] < 1e-7)
			eigenvalues[i] = 1.;

	mPreIn = (eigenvectors.array().rowwise() * eigenvalues.sqrt().cwiseInverse()).matrix()
		* eigenvectors.transpose();
	mPreInInv = (eigenvectors.array().rowwise() * eigenvalues.sqrt()).matrix()
		* eigenvectors.transpose();

	mMeanOut = VectorXd::Zero(dimOut);
	mPreOut = MatrixXd::Identity(dimOut, dimOut);
	mPreOutInv = MatrixXd::Identity(dimOut, dimOut);
	mPredictor = MatrixXd::Zero(dimOut, input.rows());
	mGradTransform = MatrixXd::Zero(dimOut, input.rows());
	mLogJacobian = 1.;
}
