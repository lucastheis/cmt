#include "utils.h"
#include "pcatransform.h"

#include "Eigen/Eigenvalues"
using Eigen::SelfAdjointEigenSolver;

MCM::PCATransform::PCATransform(const ArrayXXd& data, int numPCs) :
	MCM::LinearTransform(MatrixXd::Identity(data.rows(), data.rows()))
{
	if(numPCs < 0 || numPCs > data.rows())
		numPCs = data.rows();

	// compute eigenvectors and eigenvalues
	SelfAdjointEigenSolver<MatrixXd> eigenSolver(covariance(data));
	const MatrixXd& vecs = eigenSolver.eigenvectors();
	VectorXd vals = eigenSolver.eigenvalues();

	// store eigenvalues
	mEigenvalues = vals;

	// prevent blowing up of noise
	for(int i = data.rows() - numPCs; i < vals.size(); ++i)
		if(vals[i] < 1e-8)
			vals[i] = 1.;

	vals = vals.array().sqrt();
	mMat = vals.tail(numPCs).cwiseInverse().asDiagonal() * vecs.rightCols(numPCs).transpose();
}
