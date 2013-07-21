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

#ifdef __GXX_EXPERIMENTAL_CXX0X__
#include <random>
using std::mt19937;
using std::normal_distribution;
#endif

MatrixXd CMT::signum(const MatrixXd& matrix) {
	return (matrix.array() > 0.).cast<double>() - (matrix.array() < 0.).cast<double>();
}



#ifdef __GXX_EXPERIMENTAL_CXX0X__
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

#else // no C++11

/**
 * Based on implementation by John Cook.
 */
double CMT::gamma(double x) {
	if (x <= 0.0)
		throw Exception("Argument to gamma function must be positive.");

	// Euler's gamma constant
	const double gamma = 0.577215664901532860606512090;

	// first interval: (0, 0.001)
	if (x < 0.001)
		return 1.0 / (x * (1.0 + gamma * x));

	// second interval: [0.001, 12)
	if (x < 12.0) {
		// the algorithm directly approximates gamma over (1, 2) and uses
		// reduction identities to reduce other arguments to this interval.

		double y = x;
		int n = 0;
		bool arg_was_less_than_one = (y < 1.0);

		// add or subtract integers as necessary to bring y into (1, 2)
		// will correct for this below
		if (arg_was_less_than_one) {
			y += 1.0;
		} else {
			n = static_cast<int>(floor(y)) - 1; // will use n later
			y -= n;
		}

		// numerator coefficients for approximation over the interval (1,2)
		static const double p[] = {
			-1.71618513886549492533811E+0,
			2.47656508055759199108314E+1,
			-3.79804256470945635097577E+2,
			6.29331155312818442661052E+2,
			8.66966202790413211295064E+2,
			-3.14512729688483675254357E+4,
			-3.61444134186911729807069E+4,
			6.64561438202405440627855E+4
		};

		// denominator coefficients for approximation over the interval (1,2)
		static const double q[] = {
			-3.08402300119738975254353E+1,
			3.15350626979604161529144E+2,
			-1.01515636749021914166146E+3,
			-3.10777167157231109440444E+3,
			2.25381184209801510330112E+4,
			4.75584627752788110767815E+3,
			-1.34659959864969306392456E+5,
			-1.15132259675553483497211E+5
		};

		double num = 0.0;
		double den = 1.0;
		int i;

		double z = y - 1;
		for (i = 0; i < 8; i++) {
			num = (num + p[i])*z;
			den = den*z + q[i];
		}

		double result = num / den + 1.0;

		// apply correction if argument was not initially in (1,2)
		if (arg_was_less_than_one) {
			// use identity gamma(z) = gamma(z + 1) / z
			// the variable "result" now holds gamma of the original y + 1
			// thus we use y - 1 to get back the orginal y.
			result /= (y-1.0);
		} else {
			// use the identity gamma(z+n) = z * (z + 1) * ... * (z + n - 1) * gamma(z)
			for (i = 0; i < n; i++)
				result *= y++;
		}

		return result;
	}

	// third interval: [12, infinity)
	if (x > 171.624) {
		// Correct answer too large to display. Force +infinity.
		double temp = numeric_limits<double>::max();
		return temp*2.0;
	}

	return exp(lnGamma(x));
}



ArrayXXd CMT::gamma(const ArrayXXd& arr) {
	ArrayXXd result(arr.rows(), arr.cols());

	#pragma omp parallel for
	for(int i = 0; i < arr.size(); ++i)
		result(i) = gamma(arr(i));

	return result;
}



/**
 * Based on implementation by John Cook.
 */
double CMT::lnGamma(double x) {
	if (x <= 0.0)
		throw Exception("Argument to gamma function must be positive.");

	if (x < 12.0)
		return log(fabs(gamma(x)));

	// Abramowitz and Stegun 6.1.41
	// Asymptotic series should be good to at least 11 or 12 figures
	// For error analysis, see Whittiker and Watson
	// A Course in Modern Analysis (1927), page 252

	static const double halfLogTwoPi = 0.91893853320467274178032973640562;
	static const double c[8] = {
		 1.0 / 12.0,
		-1.0 / 360.0,
		 1.0 / 1260.0,
		-1.0 / 1680.0,
		 1.0 / 1188.0,
		-691.0 / 360360.0,
		 1.0 / 156.0,
		-3617.0 / 122400.0
	};

	double z = 1.0/(x*x);
	double sum = c[7];

	for (int i = 6; i >= 0; --i) {
		sum *= z;
		sum += c[i];
	}

	double series = sum / x;

	return (x - 0.5) * log(x) - x + halfLogTwoPi + series;
}



ArrayXXd CMT::lnGamma(const ArrayXXd& arr) {
	ArrayXXd result(arr.rows(), arr.cols());

	#pragma omp parallel for
	for(int i = 0; i < arr.size(); ++i)
		result(i) = lnGamma(arr(i));

	return result;
}
#endif



Array<double, 1, Dynamic> CMT::logSumExp(const ArrayXXd& array) {
	Array<double, 1, Dynamic> arrayMax = array.colwise().maxCoeff() - 1.;
	return arrayMax + (array.rowwise() - arrayMax).exp().colwise().sum().log();
}



Array<double, 1, Dynamic> CMT::logMeanExp(const ArrayXXd& array) {
	Array<double, 1, Dynamic> arrayMax = array.colwise().maxCoeff() - 1.;
	return arrayMax + (array.rowwise() - arrayMax).exp().colwise().mean().log();
}



#ifdef __GXX_EXPERIMENTAL_CXX0X__
ArrayXXd CMT::sampleNormal(int m, int n) {
	mt19937 gen(rand());
	normal_distribution<double> normal;
	ArrayXXd samples(m, n);

	for(int i = 0; i < samples.size(); ++i)
		samples(i) = normal(gen);

	return samples;
}
#else
ArrayXXd CMT::sampleNormal(int m, int n) {
	ArrayXXd U = ArrayXXd::Random(m, n);
	ArrayXXd V = ArrayXXd::Random(m, n);
	ArrayXXd S = U.square() + V.square();

	for(int i = 0; i < S.size(); ++i)
		while(S(i) == 0. || S(i) > 1.) {
			U(i) = ArrayXXd::Random(1, 1)(0);
			V(i) = ArrayXXd::Random(1, 1)(0);
			S(i) = U(i) * U(i) + V(i) * V(i);
		}

	// Box-Muller transform
	return U * (-2. * S.log() / S).sqrt();
}
#endif



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
	ArrayXXi samples = ArrayXXi::Zero(m, n);
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
	ArrayXXi samples = ArrayXXi::Zero(lambda.rows(), lambda.cols());
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



set<int> CMT::randomSelect(int k, int n) {
	if(k > n)
		throw Exception("k must be smaller than n.");
	if(k < 1 || n < 1)
		throw Exception("n and k must be positive.");

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
