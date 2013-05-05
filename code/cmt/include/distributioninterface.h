#ifndef DISTRIBUTIONINTERFACE_H
#define DISTRIBUTIONINTERFACE_H

#define PY_ARRAY_UNIQUE_SYMBOL CMT_ARRAY_API
#define NO_IMPORT_ARRAY

#include <Python.h>
#include <arrayobject.h>
#include "pyutils.h"

#include "distribution.h"
using CMT::Distribution;

struct DistributionObject {
	PyObject_HEAD
	Distribution* dist;
	bool owner;
};

extern const char* Distribution_doc;
extern const char* Distribution_sample_doc;
extern const char* Distribution_loglikelihood_doc;
extern const char* Distribution_evaluate_doc;

PyObject* Distribution_new(PyTypeObject*, PyObject*, PyObject*);
int Distribution_init(DistributionObject*, PyObject*, PyObject*);
void Distribution_dealloc(DistributionObject*);

PyObject* Distribution_dim(DistributionObject*, PyObject*, void*);

PyObject* Distribution_sample(DistributionObject*, PyObject*, PyObject*);
PyObject* Distribution_loglikelihood(DistributionObject*, PyObject*, PyObject*);
PyObject* Distribution_evaluate(DistributionObject*, PyObject*, PyObject*);

#endif
