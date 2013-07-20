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

extern PyTypeObject Nonlinearity_type;

extern const char* Nonlinearity_doc;
extern const char* LogisticFunction_doc;
extern const char* LogisticFunction_reduce_doc;

PyObject* Nonlinearity_new(PyTypeObject* type, PyObject* args, PyObject* kwds);
int Nonlinearity_init(NonlinearityObject*, PyObject*, PyObject*);
void Nonlinearity_dealloc(NonlinearityObject*);
PyObject* Nonlinearity_call(NonlinearityObject*, PyObject*, PyObject*);

int LogisticFunction_init(LogisticFunctionObject*, PyObject*, PyObject*);
PyObject* LogisticFunction_reduce(LogisticFunctionObject*, PyObject*);

#endif
