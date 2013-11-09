#ifndef TRAINABLEINTERFACE_H
#define TRAINABLEINTERFACE_H

#define PY_ARRAY_UNIQUE_SYMBOL CMT_ARRAY_API
#define NO_IMPORT_ARRAY

#include <Python.h>
#include <arrayobject.h>
#include "pyutils.h"
#include "callbackinterface.h"

#include "trainable.h"
using CMT::Trainable;

struct TrainableObject {
	PyObject_HEAD
	Trainable* distribution;
	bool owner;
};

extern const char* Trainable_initialize_doc;
extern const char* Trainable_parameters_doc;
extern const char* Trainable_set_parameters_doc;
extern const char* Trainable_parameter_gradient_doc;
extern const char* Trainable_fisher_information_doc;
extern const char* Trainable_check_gradient_doc;
extern const char* Trainable_check_performance_doc;

Trainable::Parameters* PyObject_ToParameters(
	PyObject* parameters,
	Trainable::Parameters* params);
Trainable::Parameters* PyObject_ToParameters(PyObject* parameters);

PyObject* Trainable_initialize(TrainableObject* self, PyObject* args, PyObject* kwds);

PyObject* Trainable_train(
	TrainableObject* self,
	PyObject* args,
	PyObject* kwds,
	Trainable::Parameters* (*PyObject_ToParameters)(PyObject*));

PyObject* Trainable_parameters(
	TrainableObject* self,
	PyObject* args,
	PyObject* kwds,
	Trainable::Parameters* (*PyObject_ToParameters)(PyObject*));

PyObject* Trainable_set_parameters(
	TrainableObject* self,
	PyObject* args,
	PyObject* kwds,
	Trainable::Parameters* (*PyObject_ToParameters)(PyObject*));

PyObject* Trainable_check_gradient(
	TrainableObject* self,
	PyObject* args,
	PyObject* kwds,
	Trainable::Parameters* (*PyObject_ToParameters)(PyObject*));

PyObject* Trainable_parameter_gradient(
	TrainableObject* self,
	PyObject* args,
	PyObject* kwds,
	Trainable::Parameters* (*PyObject_ToParameters)(PyObject*));

PyObject* Trainable_fisher_information(
	TrainableObject* self,
	PyObject* args,
	PyObject* kwds,
	Trainable::Parameters* (*PyObject_ToParameters)(PyObject*));

PyObject* Trainable_check_performance(
	TrainableObject* self,
	PyObject* args,
	PyObject* kwds,
	Trainable::Parameters* (*PyObject_ToParameters)(PyObject*));

#endif
