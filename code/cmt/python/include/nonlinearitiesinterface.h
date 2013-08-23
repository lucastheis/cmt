#ifndef NONLINEARITIESINTERFACE_H
#define NONLINEARITIESINTERFACE_H

#define PY_ARRAY_UNIQUE_SYMBOL CMT_ARRAY_API
#define NO_IMPORT_ARRAY

#include <Python.h>
#include <arrayobject.h>
#include "pyutils.h"

#include "cmt/nonlinear"
using CMT::Nonlinearity;
using CMT::LogisticFunction;
using CMT::ExponentialFunction;
using CMT::HistogramNonlinearity;

struct NonlinearityObject {
	PyObject_HEAD
	Nonlinearity* nonlinearity;
	bool owner;
};

struct LogisticFunctionObject {
	PyObject_HEAD
	LogisticFunction* nonlinearity;
	bool owner;
};

struct ExponentialFunctionObject {
	PyObject_HEAD
	ExponentialFunction* nonlinearity;
	bool owner;
};

struct HistogramNonlinearityObject {
	PyObject_HEAD
	HistogramNonlinearity* nonlinearity;
	bool owner;
};

extern PyTypeObject Nonlinearity_type;

extern const char* Nonlinearity_doc;
extern const char* Nonlinearity_reduce_doc;
extern const char* LogisticFunction_doc;
extern const char* ExponentialFunction_doc;
extern const char* HistogramNonlinearity_doc;

PyObject* Nonlinearity_new(PyTypeObject* type, PyObject* args, PyObject* kwds);
int Nonlinearity_init(NonlinearityObject*, PyObject*, PyObject*);
void Nonlinearity_dealloc(NonlinearityObject*);
PyObject* Nonlinearity_call(NonlinearityObject*, PyObject*, PyObject*);
PyObject* Nonlinearity_reduce(NonlinearityObject*, PyObject*);

int LogisticFunction_init(LogisticFunctionObject*, PyObject*, PyObject*);
int ExponentialFunction_init(ExponentialFunctionObject*, PyObject*, PyObject*);
int HistogramNonlinearity_init(HistogramNonlinearityObject*, PyObject*, PyObject*);

#endif
