#ifndef UTILS_H
#define UTILS_H

#include "Eigen/Core"
#include <vector>

using namespace Eigen;
using std::vector;

#define PI 3.141592653589793

Array<double, 1, Dynamic> logsumexp(const ArrayXXd& array);
Array<double, 1, Dynamic> logmeanexp(const ArrayXXd& array);

ArrayXXd sampleNormal(int m = 1, int n = 1);
ArrayXXd sampleGamma(int m = 1, int n = 1, int k = 1);

VectorXi argsort(const VectorXd& data);
MatrixXd covariance(const MatrixXd& data);
MatrixXd corrcoef(const MatrixXd& data);
MatrixXd normalize(const MatrixXd& matrix);

double logDetPD(const MatrixXd& matrix);

MatrixXd deleteRows(const MatrixXd& matrix, vector<int> indices);
MatrixXd deleteCols(const MatrixXd& matrix, vector<int> indices);

#endif
