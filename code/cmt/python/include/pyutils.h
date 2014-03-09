#ifndef PYUTILS_H
#define PYUTILS_H

#define PY_ARRAY_UNIQUE_SYMBOL CMT_ARRAY_API
#define NO_IMPORT_ARRAY

#include <Python.h>
#include <arrayobject.h>

#include <vector>
using std::vector;

#include "Eigen/Core"
using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::Array;
using Eigen::ArrayXXd;
using Eigen::Dynamic;

#include "cmt/tools"
using CMT::Tuples;

#include "cmt/utils"
using CMT::Regularizer;

typedef Matrix<bool, Dynamic, Dynamic> MatrixXb;
typedef Array<bool, Dynamic, Dynamic> ArrayXXb;

PyObject* PyArray_FromMatrixXd(const MatrixXd& mat);
PyObject* PyArray_FromMatrixXi(const MatrixXi& mat);
PyObject* PyArray_FromMatrixXb(const MatrixXb& mat);
MatrixXd PyArray_ToMatrixXd(PyObject* array);
MatrixXi PyArray_ToMatrixXi(PyObject* array);
MatrixXb PyArray_ToMatrixXb(PyObject* array);
vector<ArrayXXd> PyArray_ToArraysXXd(PyObject* array);
vector<ArrayXXb> PyArray_ToArraysXXb(PyObject* array);
PyObject* PyArray_FromArraysXXd(const vector<ArrayXXd>& channels);

Tuples PyList_AsTuples(PyObject* list);
PyObject* PyList_FromTuples(const Tuples& tuples);

Regularizer PyObject_ToRegularizer(PyObject* regularizer);

#endif
