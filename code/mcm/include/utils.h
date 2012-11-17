#ifndef UTILS_H
#define UTILS_H

#include "Eigen/Core"
using Eigen::Array;
using Eigen::ArrayXXd;
using Eigen::Dynamic;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;

#include <vector>
using std::vector;

#include <set>
using std::set;

#define PI 3.141592653589793

Array<double, 1, Dynamic> logSumExp(const ArrayXXd& array);
Array<double, 1, Dynamic> logMeanExp(const ArrayXXd& array);

ArrayXXd sampleNormal(int m = 1, int n = 1);
ArrayXXd sampleGamma(int m = 1, int n = 1, int k = 1);
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

#endif
