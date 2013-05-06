#ifndef TRAINABLEINTERFACE_H
#define TRAINABLEINTERFACE_H

#define PY_ARRAY_UNIQUE_SYMBOL CMT_ARRAY_API
#define NO_IMPORT_ARRAY

#include <Python.h>
#include <arrayobject.h>
#include "pyutils.h"

#include "trainable.h"
using CMT::Trainable;

struct TrainableObject {
	PyObject_HEAD
	Trainable* distribution;
	bool owner;
};

extern const char* Trainable_parameters_doc;
extern const char* Trainable_set_parameters_doc;

Trainable::Parameters* PyObject_ToParameters(PyObject* parameters);

PyObject* Trainable_train(
	TrainableObject* self,
	PyObject* args,
	PyObject* kwds,
	Trainable::Parameters* (*PyObject_ToParameters)(PyObject*) = &PyObject_ToParameters);

PyObject* Trainable_parameters(
	TrainableObject* self,
	PyObject* args,
	PyObject* kwds,
	Trainable::Parameters* (*PyObject_ToParameters)(PyObject*) = &PyObject_ToParameters);

PyObject* Trainable_set_parameters(
	TrainableObject* self,
	PyObject* args,
	PyObject* kwds,
	Trainable::Parameters* (*PyObject_ToParameters)(PyObject*) = &PyObject_ToParameters);

PyObject* Trainable_check_gradient(
	TrainableObject* self,
	PyObject* args,
	PyObject* kwds,
	Trainable::Parameters* (*PyObject_ToParameters)(PyObject*) = &PyObject_ToParameters);

PyObject* Trainable_parameter_gradient(
	TrainableObject* self,
	PyObject* args,
	PyObject* kwds,
	Trainable::Parameters* (*PyObject_ToParameters)(PyObject*) = &PyObject_ToParameters);

#endif
