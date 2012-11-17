#include "Eigen/Cholesky"
#include "Eigen/SVD"
#include "utils.h"
#include "exception.h"
#include <algorithm>
#include <vector>
#include <iostream>
#include <cstdlib>

#ifdef __GXX_EXPERIMENTAL_CXX0X__
#include <random>
#endif

using namespace Eigen;
using namespace std;

Array<double, 1, Dynamic> logSumExp(const ArrayXXd& array) {
	Array<double, 1, Dynamic> arrayMax = array.colwise().maxCoeff() - 1.;
	return arrayMax + (array.rowwise() - arrayMax).exp().colwise().sum().log();
}



Array<double, 1, Dynamic> logMeanExp(const ArrayXXd& array) {
	Array<double, 1, Dynamic> arrayMax = array.colwise().maxCoeff() - 1.;
	return arrayMax + (array.rowwise() - arrayMax).exp().colwise().mean().log();
}



#ifdef __GXX_EXPERIMENTAL_CXX0X__
ArrayXXd sampleNormal(int m, int n) {
	mt19937 gen(rand());
	normal_distribution<double> normal;
	ArrayXXd samples(m, n);

	for(int i = 0; i < samples.size(); ++i)
		samples(i) = normal(gen);

	return samples;
}
#else
#warning "No C++11 support. Using my own implementation of the Box-Muller transform."
ArrayXXd sampleNormal(int m, int n) {
	ArrayXXd U1 = ArrayXXd::Random(m, n).abs();
	ArrayXXd U2 = ArrayXXd::Random(m, n).abs();
	// Box-Muller transform
	return (-2. * U1.log()).sqrt() * (2. * PI * U2).cos();
}
#endif



ArrayXXd sampleGamma(int m, int n, int k) {
	ArrayXXd samples = ArrayXXd::Zero(m, n);

	for(int i = 0; i < k; ++i)
		samples -= ArrayXXd::Random(m, n).abs().log();

	return samples;
}



VectorXi argSort(const VectorXd& data) {
	// create pairs of values and indices
	vector<pair<double, int> > pairs(data.size());
	for(int i = 0; i < data.size(); ++i) {
		pairs[i].first = data[i];
		pairs[i].second = i;
	}

	// sort values in descending order
	sort(pairs.begin(), pairs.end(), greater<pair<double, int> >());

	// store indices
	VectorXi indices(data.size());
	for(int i = 0; i < data.size(); ++i)
		indices[pairs[i].second] = i;

	return indices;
}



MatrixXd covariance(const MatrixXd& data) {
	MatrixXd data_centered = data.colwise() - data.rowwise().mean().eval();
	return data_centered * data_centered.transpose() / data.cols();
}



MatrixXd covariance(const MatrixXd& input, const MatrixXd& output) {
	if(input.cols() != output.cols())
		throw Exception("Number of inputs and outputs must be the same.");

	MatrixXd input_centered = input.colwise() - input.rowwise().mean().eval();
	MatrixXd output_centered = output.colwise() - output.rowwise().mean().eval();
	return input_centered * output_centered.transpose() / output.cols();
}



MatrixXd corrCoef(const MatrixXd& data) {
	MatrixXd C = covariance(data);
	VectorXd c = C.diagonal();
	return C.array() / (c * c.transpose()).array().sqrt();
}



MatrixXd normalize(const MatrixXd& matrix) {
	return matrix.array().rowwise() / matrix.colwise().norm().eval().array();
}



MatrixXd pInverse(const MatrixXd& matrix) {
	JacobiSVD<MatrixXd> svd(matrix, ComputeThinU | ComputeThinV);

	VectorXd svInv = svd.singularValues();

	for(int i = 0; i < svInv.size(); ++i)
		if(svInv[i] > 1e-8)
			svInv[i] = 1. / svInv[i];

	return svd.matrixV() * svInv.asDiagonal() * svd.matrixU().transpose();
}



double logDetPD(const MatrixXd& matrix) {
	return 2. * matrix.llt().matrixLLT().diagonal().array().log().sum();
}



MatrixXd deleteRows(const MatrixXd& matrix, vector<int> indices) {
	MatrixXd result = ArrayXXd::Zero(matrix.rows() - indices.size(), matrix.cols());

	sort(indices.begin(), indices.end());

	unsigned int idx = 0;

	for(int i = 0; i < matrix.rows(); ++i) {
		if(idx < indices.size() && indices[idx] == i) {
			++idx;
			continue;
		}
		result.row(i - idx) = matrix.row(i);
	}

	return result;
}



MatrixXd deleteCols(const MatrixXd& matrix, vector<int> indices) {
	MatrixXd result = ArrayXXd::Zero(matrix.rows(), matrix.cols() - indices.size());

	sort(indices.begin(), indices.end());

	unsigned int idx = 0;

	for(int i = 0; i < matrix.cols(); ++i) {
		if(idx < indices.size() && indices[idx] == i) {
			++idx;
			continue;
		}
		result.col(i - idx) = matrix.col(i);
	}

	return result;
}
