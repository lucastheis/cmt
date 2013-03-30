#ifndef MCBMINTERFACE_H
#define MCBMINTERFACE_H

#define PY_ARRAY_UNIQUE_SYMBOL CMT_ARRAY_API
#define NO_IMPORT_ARRAY

#include <Python.h>
#include <arrayobject.h>
#include "pyutils.h"
#include "mcbm.h"

struct MCBMObject {
	PyObject_HEAD
	MCBM* mcbm;
};

extern PyTypeObject MCBM_type;

extern const char* MCBM_doc;
extern const char* MCBM_sample_doc;
extern const char* MCBM_loglikelihood_doc;
extern const char* MCBM_evaluate_doc;
extern const char* MCBM_train_doc;
extern const char* MCBM_parameters_doc;
extern const char* MCBM_set_parameters_doc;
extern const char* MCBM_reduce_doc;
extern const char* MCBM_setstate_doc;

PyObject* MCBM_new(PyTypeObject*, PyObject*, PyObject*);
int MCBM_init(MCBMObject*, PyObject*, PyObject*);
void MCBM_dealloc(MCBMObject*);

PyObject* MCBM_dim_in(MCBMObject*, PyObject*, void*);
PyObject* MCBM_dim_out(MCBMObject*, PyObject*, void*);

PyObject* MCBM_num_components(MCBMObject*, PyObject*, void*);
PyObject* MCBM_num_features(MCBMObject*, PyObject*, void*);

PyObject* MCBM_priors(MCBMObject*, PyObject*, void*);
int MCBM_set_priors(MCBMObject*, PyObject*, void*);

PyObject* MCBM_weights(MCBMObject*, PyObject*, void*);
int MCBM_set_weights(MCBMObject*, PyObject*, void*);

PyObject* MCBM_features(MCBMObject*, PyObject*, void*);
int MCBM_set_features(MCBMObject*, PyObject*, void*);

PyObject* MCBM_predictors(MCBMObject*, PyObject*, void*);
int MCBM_set_predictors(MCBMObject*, PyObject*, void*);

PyObject* MCBM_input_bias(MCBMObject*, PyObject*, void*);
int MCBM_set_input_bias(MCBMObject*, PyObject*, void*);

PyObject* MCBM_output_bias(MCBMObject*, PyObject*, void*);
int MCBM_set_output_bias(MCBMObject*, PyObject*, void*);

PyObject* MCBM_sample(MCBMObject*, PyObject*, PyObject*);
PyObject* MCBM_loglikelihood(MCBMObject*, PyObject*, PyObject*);
PyObject* MCBM_evaluate(MCBMObject*, PyObject*, PyObject*);

PyObject* MCBM_train(MCBMObject*, PyObject*, PyObject*);

PyObject* MCBM_parameters(MCBMObject*, PyObject*, PyObject*);
PyObject* MCBM_set_parameters(MCBMObject*, PyObject*, PyObject*);
PyObject* MCBM_compute_gradient(MCBMObject*, PyObject*, PyObject*);
PyObject* MCBM_check_gradient(MCBMObject*, PyObject*, PyObject*);

PyObject* MCBM_reduce(MCBMObject*, PyObject*, PyObject*);
PyObject* MCBM_setstate(MCBMObject*, PyObject*, PyObject*);

#endif
