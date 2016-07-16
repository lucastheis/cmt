#ifndef CMT_UTILS_H
#define CMT_UTILS_H

#include "Eigen/Core"
#include <vector>
#include <set>
#include "exception.h"

#define PI 3.141592653589793

namespace CMT {
	using Eigen::Array;
	using Eigen::ArrayXXd;
	using Eigen::ArrayXXi;
	using Eigen::Dynamic;
	using Eigen::MatrixXd;
	using Eigen::MatrixXi;
	using Eigen::VectorXd;
	using Eigen::VectorXi;

	using std::vector;
	using std::set;

	Array<double, 1, Dynamic> logSumExp(const ArrayXXd& array);
	Array<double, 1, Dynamic> logMeanExp(const ArrayXXd& array);

	MatrixXd signum(const MatrixXd& matrix);

	double gamma(double x);
	double lnGamma(double x);
	ArrayXXd gamma(const ArrayXXd& arr);
	ArrayXXd lnGamma(const ArrayXXd& arr);

	ArrayXXd tanh(const ArrayXXd& arr);
	ArrayXXd cosh(const ArrayXXd& arr);
	ArrayXXd sinh(const ArrayXXd& arr);
	ArrayXXd sech(const ArrayXXd& arr);

	ArrayXXd sampleNormal(int m = 1, int n = 1);
	ArrayXXd sampleGamma(int m = 1, int n = 1, int k = 1);
	ArrayXXi samplePoisson(int m = 1, int n = 1, double lambda = 1.);
	ArrayXXi samplePoisson(const ArrayXXd& lambda);
	ArrayXXi sampleBinomial(int w = 1, int h = 1, int n = 10, double p = .5);
	ArrayXXi sampleBinomial(const ArrayXXi& n, const ArrayXXd& p);
	set<int> randomSelect(int k, int n);

	VectorXi argSort(const VectorXd& data);
	MatrixXd covariance(const MatrixXd& data);
	MatrixXd covariance(const MatrixXd& input, const MatrixXd& output);
	MatrixXd corrCoef(const MatrixXd& data);
	MatrixXd normalize(const MatrixXd& matrix);
	MatrixXd pInverse(const MatrixXd& matrix);

	double logDetPD(const MatrixXd& matrix);

	MatrixXd deleteRows(const MatrixXd& matrix, vector<int> indices);
	MatrixXd deleteCols(const MatrixXd& matrix, vector<int> indices);

	template <class ArrayType>
	ArrayType concatenate(const vector<ArrayType>& data, int axis=1);
}



template <class ArrayType>
ArrayType CMT::concatenate(const vector<ArrayType>& data, int axis) {
	if(data.size()) {
		if(axis == 1) {
			int cols = 0;
			int rows = data[0].rows();

			for(int i = 0; i < data.size(); ++i) {
				cols += data[i].cols();
				
				if(data[i].rows() != rows)
					throw Exception("Arrays must have the same number of rows for concatenation.");
			}

			// concatenate horizontally
			ArrayType result(rows, cols);
			for(int col = 0, i = 0; i < data.size(); col += data[i].cols(), ++i)
				result.middleCols(col, data[i].cols()) = data[i];

			return result;
		} else if(axis == 0) {
			int cols = data[0].cols();
			int rows = 0;

			for(int i = 0; i < data.size(); ++i) {
				rows += data[i].rows();
				
				if(data[i].cols() != cols)
					throw Exception("Arrays must have the same number of columns for concatenation.");
			}

			// concatenate horizontally
			ArrayType result(rows, cols);
			for(int row = 0, i = 0; i < data.size(); row += data[i].rows(), ++i)
				result.middleRows(row, data[i].rows()) = data[i];

			return result;
		} else {
			throw Exception("Axis should be 0 or 1.");
		}
	} else {
		return ArrayType();
	}
}

#endif
