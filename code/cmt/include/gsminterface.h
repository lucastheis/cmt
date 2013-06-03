#ifndef GSMINTERFACE_H
#define GSMINTERFACE_H

#define PY_ARRAY_UNIQUE_SYMBOL CMT_ARRAY_API
#define NO_IMPORT_ARRAY

#include <Python.h>
#include <arrayobject.h>
#include "pyutils.h"

#include "gsm.h"
using CMT::GSM;

struct GSMObject {
	PyObject_HEAD
	GSM* gsm;
	bool owner;
};

int GSM_init(GSMObject*, PyObject*, PyObject*);

PyObject* GSM_mean(GSMObject*, void*);
int GSM_set_mean(GSMObject*, PyObject*, void*);

PyObject* GSM_priors(GSMObject*, void*);
int GSM_set_priors(GSMObject*, PyObject*, void*);

PyObject* GSM_scales(GSMObject*, void*);
int GSM_set_scales(GSMObject*, PyObject*, void*);

PyObject* GSM_covariance(GSMObject*, void*);
int GSM_set_covariance(GSMObject*, PyObject*, void*);

PyObject* GSM_reduce(GSMObject*, PyObject*);
PyObject* GSM_setstate(GSMObject*, PyObject*);

#endif
