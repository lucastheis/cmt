#ifndef PYUTILS_H
#define PYUTILS_H

#define PY_ARRAY_UNIQUE_SYMBOL MCM_ARRAY_API
#define NO_IMPORT_ARRAY

#include <Python.h>
#include <arrayobject.h>
#include <vector>
#include "Eigen/Core"

using namespace Eigen;
using std::vector;

typedef Matrix<bool, Dynamic, Dynamic> MatrixXb;

PyObject* PyArray_FromMatrixXd(const MatrixXd& mat);
PyObject* PyArray_FromMatrixXi(const MatrixXi& mat);
PyObject* PyArray_FromMatrixXb(const MatrixXb& mat);
MatrixXd PyArray_ToMatrixXd(PyObject* array);
MatrixXi PyArray_ToMatrixXi(PyObject* array);
MatrixXb PyArray_ToMatrixXb(PyObject* array);
vector<ArrayXXd> PyArray_ToArraysXXd(PyObject* array);
PyObject* PyArray_FromArraysXXd(const vector<ArrayXXd>& channels);

#endif
