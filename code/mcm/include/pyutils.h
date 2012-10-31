#ifndef PYUTILS_H
#define PYUTILS_H

#define PY_ARRAY_UNIQUE_SYMBOL MCM_ARRAY_API
#define NO_IMPORT_ARRAY

#include <Python.h>
#include <arrayobject.h>
#include "Eigen/Core"

using namespace Eigen;

typedef Matrix<bool, Dynamic, Dynamic> MatrixXb;

PyObject* PyArray_FromMatrixXd(const MatrixXd& mat);
PyObject* PyArray_FromMatrixXi(const MatrixXi& mat);
PyObject* PyArray_FromMatrixXb(const MatrixXb& mat);
MatrixXd PyArray_ToMatrixXd(PyObject* array);
MatrixXi PyArray_ToMatrixXi(PyObject* array);
MatrixXb PyArray_ToMatrixXb(PyObject* array);

#endif
