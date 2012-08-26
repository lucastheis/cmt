#ifndef PYUTILS_H
#define PYUTILS_H

#define PY_ARRAY_UNIQUE_SYMBOL MCM_ARRAY_API
#define NO_IMPORT_ARRAY

#include <Python.h>
#include <arrayobject.h>
#include "Eigen/Core"

using namespace Eigen;

PyObject* PyArray_FromMatrixXd(const MatrixXd& mat);
MatrixXd PyArray_ToMatrixXd(PyObject* array);

#endif
