#include "utils.h"
#include <cstdlib>

#include "Eigen/Core"
using Eigen::Dynamic;
using Eigen::Array;
using Eigen::ArrayXXd;
using Eigen::ArrayXXi;
using Eigen::MatrixXd;
using Eigen::VectorXi;

#include "Eigen/SVD"
using Eigen::JacobiSVD;
using Eigen::ComputeThinU;
using Eigen::ComputeThinV;

#include <cmath>
using std::exp;
using std::log;
using std::floor;
using std::tanh;
using std::sinh;
using std::cosh;
#ifdef __GXX_EXPERIMENTAL_CXX0X__
using std::lgamma;
using std::tgamma;
#endif

#include <cstdlib>
using std::rand;

#include <set>
using std::set;
using std::pair;

#include <algorithm>
using std::greater;
using std::sort;

#include <limits>
using std::numeric_limits;

#include <random>
using std::mt19937;
using std::normal_distribution;

MatrixXd CMT::signum(const MatrixXd& matrix) {
	return (matrix.array() > 0.).cast<double>() - (matrix.array() < 0.).cast<double>();
}



double CMT::gamma(double x) {
	if (x <= 0.0)
		throw Exception("Argument to gamma function must be positive.");
	return tgamma(x);
}



ArrayXXd CMT::gamma(const ArrayXXd& arr) {
	ArrayXXd result(arr.rows(), arr.cols());

	#pragma omp parallel for
	for(int i = 0; i < arr.size(); ++i)
		result(i) = tgamma(arr(i));

	return result;
}



double CMT::lnGamma(double x) {
	if (x <= 0.0)
		throw Exception("Argument to gamma function must be positive.");
	return lgamma(x);
}



ArrayXXd CMT::lnGamma(const ArrayXXd& arr) {
	ArrayXXd result(arr.rows(), arr.cols());

	#pragma omp parallel for
	for(int i = 0; i < arr.size(); ++i)
		result(i) = lgamma(arr(i));

	return result;
}



ArrayXXd CMT::tanh(const ArrayXXd& arr) {
	ArrayXXd result(arr.rows(), arr.cols());

	#pragma omp parallel for
	for(int i = 0; i < arr.size(); ++i)
		result(i) = std::tanh(arr(i));

	return result;
}



ArrayXXd CMT::cosh(const ArrayXXd& arr) {
	ArrayXXd result(arr.rows(), arr.cols());

	#pragma omp parallel for
	for(int i = 0; i < arr.size(); ++i)
		result(i) = std::cosh(arr(i));

	return result;
}



ArrayXXd CMT::sinh(const ArrayXXd& arr) {
	ArrayXXd result(arr.rows(), arr.cols());

	#pragma omp parallel for
	for(int i = 0; i < arr.size(); ++i)
		result(i) = std::sinh(arr(i));

	return result;
}



ArrayXXd CMT::sech(const ArrayXXd& arr) {
	ArrayXXd result(arr.rows(), arr.cols());

	#pragma omp parallel for
	for(int i = 0; i < arr.size(); ++i)
		result(i) = 1. / std::cosh(arr(i));

	return result;
}



Array<double, 1, Dynamic> CMT::logSumExp(const ArrayXXd& array) {
	Array<double, 1, Dynamic> arrayMax = array.colwise().maxCoeff() - 1.;
	return arrayMax + (array.rowwise() - arrayMax).exp().colwise().sum().log();
}



Array<double, 1, Dynamic> CMT::logMeanExp(const ArrayXXd& array) {
	Array<double, 1, Dynamic> arrayMax = array.colwise().maxCoeff() - 1.;
	return arrayMax + (array.rowwise() - arrayMax).exp().colwise().mean().log();
}



ArrayXXd CMT::sampleNormal(int m, int n) {
 	static mt19937 gen(rand());

	normal_distribution<double> normal;
	ArrayXXd samples(m, n);

	for(int i = 0; i < samples.size(); ++i)
		samples(i) = normal(gen);

	return samples;
}



ArrayXXd CMT::sampleGamma(int m, int n, int k) {
	ArrayXXd samples = ArrayXXd::Zero(m, n);

	for(int i = 0; i < k; ++i)
		samples -= ArrayXXd::Random(m, n).abs().log();

	return samples;
}



/**
 * Algorithm due to Knuth, 1969.
 */
ArrayXXi CMT::samplePoisson(int m, int n, double lambda) {
	ArrayXXi samples(m, n);
	double threshold = exp(-lambda);

	#pragma omp parallel for
	for(int i = 0; i < samples.size(); ++i) {
		double p = rand() / static_cast<double>(RAND_MAX);
		int k = 0;

		while(p > threshold) {
			p *= rand() / static_cast<double>(RAND_MAX);
			k += 1;
		}

		samples(i) = k;
	}

	return samples;
}



/**
 * Algorithm due to Knuth, 1969.
 */
ArrayXXi CMT::samplePoisson(const ArrayXXd& lambda) {
	ArrayXXi samples(lambda.rows(), lambda.cols());
	ArrayXXd threshold = (-lambda).exp();

	#pragma omp parallel for
	for(int i = 0; i < samples.size(); ++i) {
		double p = rand() / static_cast<double>(RAND_MAX);
		int k = 0;

		while(p > threshold(i)) {
			k += 1;
			p *= rand() / static_cast<double>(RAND_MAX);
		}

		samples(i) = k;
	}

	return samples;
}



ArrayXXi CMT::sampleBinomial(int w, int h, int n, double p) {
	ArrayXXi samples = ArrayXXi::Zero(w, h);

	#pragma omp parallel for
	for(int i = 0; i < samples.size(); ++i) {
		// very naive algorithm for generating binomial samples
		for(int k = 0; k < n; ++k)
			if(rand() / static_cast<double>(RAND_MAX) < p)
				samples(i) += 1; 
	}

	return samples;
}



ArrayXXi CMT::sampleBinomial(const ArrayXXi& n, const ArrayXXd& p) {
	if(n.rows() != p.rows() || n.cols() != p.cols())
		throw Exception("n and p must be of the same size.");

	ArrayXXi samples = ArrayXXi::Zero(n.rows(), n.cols());

	#pragma omp parallel for
	for(int i = 0; i < samples.size(); ++i) {
		// very naive algorithm for generating binomial samples
		for(int k = 0; k < n(i); ++k)
			if(rand() / static_cast<double>(RAND_MAX) < p(i))
				samples(i) += 1; 
	}

	return samples;
}



set<int> CMT::randomSelect(int k, int n) {
	if(k > n)
		throw Exception("k must be smaller than n.");
	if(k < 0 || n < 0)
		throw Exception("n and k must be non-negative.");

	// TODO: a hash map could be more efficient
	set<int> indices;

	if(k <= n / 2) {
		for(int i = 0; i < k; ++i)
			while(indices.insert(rand() % n).second != true) {
				// repeat until insertion successful
			}
	} else {
		// fill set with all indices
		for(int i = 0; i < n; ++i)
			indices.insert(i);
		for(int i = 0; i < n - k; ++i)
			while(!indices.erase(rand() % n)) {
				// repeat until deletion successful
			}
	}

	return indices;
}



VectorXi CMT::argSort(const VectorXd& data) {
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



MatrixXd CMT::covariance(const MatrixXd& data) {
	MatrixXd dataCentered = data.colwise() - data.rowwise().mean().eval();
	return dataCentered * dataCentered.transpose() / data.cols();
}



MatrixXd CMT::covariance(const MatrixXd& input, const MatrixXd& output) {
	if(input.cols() != output.cols())
		throw Exception("Number of inputs and outputs must be the same.");

	MatrixXd inputCentered = input.colwise() - input.rowwise().mean().eval();
	MatrixXd outputCentered = output.colwise() - output.rowwise().mean().eval();
	return inputCentered * outputCentered.transpose() / output.cols();
}



MatrixXd CMT::corrCoef(const MatrixXd& data) {
	MatrixXd C = covariance(data);
	VectorXd c = C.diagonal();
	return C.array() / (c * c.transpose()).array().sqrt();
}



MatrixXd CMT::normalize(const MatrixXd& matrix) {
	return matrix.array().rowwise() / matrix.colwise().norm().eval().array();
}



MatrixXd CMT::pInverse(const MatrixXd& matrix) {
	if(matrix.size() == 0)
		return matrix.transpose();

	JacobiSVD<MatrixXd> svd(matrix, ComputeThinU | ComputeThinV);

	VectorXd svInv = svd.singularValues();

	for(int i = 0; i < svInv.size(); ++i)
		if(svInv[i] > 1e-8)
			svInv[i] = 1. / svInv[i];

	return svd.matrixV() * svInv.asDiagonal() * svd.matrixU().transpose();
}



double CMT::logDetPD(const MatrixXd& matrix) {
	return 2. * matrix.llt().matrixLLT().diagonal().array().log().sum();
}



MatrixXd CMT::deleteRows(const MatrixXd& matrix, vector<int> indices) {
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



MatrixXd CMT::deleteCols(const MatrixXd& matrix, vector<int> indices) {
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
