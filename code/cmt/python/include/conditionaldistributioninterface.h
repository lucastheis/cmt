#ifndef CONDITIONALDISTRIBUTIONINTERFACE_H
#define CONDITIONALDISTRIBUTIONINTERFACE_H

#define PY_ARRAY_UNIQUE_SYMBOL CMT_ARRAY_API
#define NO_IMPORT_ARRAY

#include <Python.h>
#include <arrayobject.h>
#include "pyutils.h"

#include "cmt/models"
using CMT::ConditionalDistribution;

struct CDObject {
	PyObject_HEAD
	ConditionalDistribution* cd;
	bool owner; 
};

extern PyTypeObject Preconditioner_type;

extern const char* CD_doc;
extern const char* CD_sample_doc;
extern const char* CD_predict_doc;
extern const char* CD_loglikelihood_doc;
extern const char* CD_evaluate_doc;

PyObject* CD_new(PyTypeObject*, PyObject*, PyObject*);
int CD_init(CDObject*, PyObject*, PyObject*);
void CD_dealloc(CDObject*);

PyObject* CD_dim_in(CDObject*, PyObject*, void*);
PyObject* CD_dim_out(CDObject*, PyObject*, void*);

PyObject* CD_sample(CDObject*, PyObject*, PyObject*);
PyObject* CD_predict(CDObject*, PyObject*, PyObject*);
PyObject* CD_loglikelihood(CDObject*, PyObject*, PyObject*);
PyObject* CD_evaluate(CDObject*, PyObject*, PyObject*);
PyObject* CD_data_gradient(CDObject*, PyObject*, PyObject*);

#endif
