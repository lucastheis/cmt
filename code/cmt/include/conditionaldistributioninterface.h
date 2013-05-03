#ifndef CMT_CONDITIONALDISTRIBUTIONINTERFACE_H
#define CMT_CONDITIONALDISTRIBUTIONINTERFACE_H

#define PY_ARRAY_UNIQUE_SYMBOL CMT_ARRAY_API
#define NO_IMPORT_ARRAY

#include <Python.h>
#include <arrayobject.h>
#include "pyutils.h"
#include "conditionaldistribution.h"

struct CDObject {
	PyObject_HEAD
	CMT::ConditionalDistribution* cd;
	bool owner; 
};

extern const char* CD_doc;
extern const char* CD_sample_doc;
extern const char* CD_loglikelihood_doc;
extern const char* CD_evaluate_doc;

PyObject* CD_new(PyTypeObject*, PyObject*, PyObject*);
int CD_init(CDObject*, PyObject*, PyObject*);
void CD_dealloc(CDObject*);

PyObject* CD_dim_in(CDObject*, PyObject*, void*);
PyObject* CD_dim_out(CDObject*, PyObject*, void*);

PyObject* CD_sample(CDObject*, PyObject*, PyObject*);
PyObject* CD_loglikelihood(CDObject*, PyObject*, PyObject*);
PyObject* CD_evaluate(CDObject*, PyObject*, PyObject*);

#endif
