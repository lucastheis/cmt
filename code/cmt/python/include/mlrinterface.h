#ifndef MLRINTERFACE_H
#define MLRINTERFACE_H

#define PY_ARRAY_UNIQUE_SYMBOL CMT_ARRAY_API
#define NO_IMPORT_ARRAY

#include <Python.h>
#include <arrayobject.h>

#include "cmt/models"
using CMT::MLR;
using CMT::Trainable;

struct MLRObject {
	PyObject_HEAD
	MLR* mlr;
	bool owner;
};

extern PyTypeObject MLR_type;

extern const char* MLR_doc;
extern const char* MLR_train_doc;
extern const char* MLR_reduce_doc;
extern const char* MLR_setstate_doc;

int MLR_init(MLRObject*, PyObject*, PyObject*);
void MLR_dealloc(MLRObject*);

PyObject* MLR_weights(MLRObject*, void*);
int MLR_set_weights(MLRObject*, PyObject*, void*);

PyObject* MLR_biases(MLRObject*, void*);
int MLR_set_biases(MLRObject*, PyObject*, void*);

PyObject* MLR_train(MLRObject*, PyObject*, PyObject*);

PyObject* MLR_parameters(MLRObject*, PyObject*, PyObject*);
PyObject* MLR_set_parameters(MLRObject*, PyObject*, PyObject*);
PyObject* MLR_parameter_gradient(MLRObject*, PyObject*, PyObject*);
PyObject* MLR_check_gradient(MLRObject*, PyObject*, PyObject*);
PyObject* MLR_check_performance(MLRObject* self, PyObject* args, PyObject* kwds);

PyObject* MLR_reduce(MLRObject*, PyObject*);
PyObject* MLR_setstate(MLRObject*, PyObject*);

Trainable::Parameters* PyObject_ToMLRParameters(PyObject* parameters);

#endif
