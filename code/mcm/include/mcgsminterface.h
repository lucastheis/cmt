#ifndef MCGSMINTERFACE_H
#define MCGSMINTERFACE_H

#define PY_ARRAY_UNIQUE_SYMBOL MCM_ARRAY_API
#define NO_IMPORT_ARRAY

#include <Python.h>
#include <arrayobject.h>
#include "mcgsm.h"
#include "pyutils.h"

struct MCGSMObject {
	PyObject_HEAD
	MCGSM* mcgsm;
};

extern PyTypeObject MCGSM_type;

PyObject* MCGSM_new(PyTypeObject*, PyObject*, PyObject*);
int MCGSM_init(MCGSMObject*, PyObject*, PyObject*);
void MCGSM_dealloc(MCGSMObject*);

PyObject* MCGSM_dim_in(MCGSMObject*, PyObject*, void*);
PyObject* MCGSM_dim_out(MCGSMObject*, PyObject*, void*);
PyObject* MCGSM_num_components(MCGSMObject*, PyObject*, void*);
PyObject* MCGSM_num_scales(MCGSMObject*, PyObject*, void*);
PyObject* MCGSM_num_features(MCGSMObject*, PyObject*, void*);

PyObject* MCGSM_priors(MCGSMObject*, PyObject*, void*);
int MCGSM_set_priors(MCGSMObject*, PyObject*, void*);

PyObject* MCGSM_scales(MCGSMObject*, PyObject*, void*);
int MCGSM_set_scales(MCGSMObject*, PyObject*, void*);

PyObject* MCGSM_weights(MCGSMObject*, PyObject*, void*);
int MCGSM_set_weights(MCGSMObject*, PyObject*, void*);

PyObject* MCGSM_features(MCGSMObject*, PyObject*, void*);
int MCGSM_set_features(MCGSMObject*, PyObject*, void*);

PyObject* MCGSM_cholesky_factors(MCGSMObject*, PyObject*, void*);
int MCGSM_set_cholesky_factors(MCGSMObject*, PyObject*, void*);

PyObject* MCGSM_predictors(MCGSMObject*, PyObject*, void*);
int MCGSM_set_predictors(MCGSMObject*, PyObject*, void*);

PyObject* MCGSM_initialize(MCGSMObject*, PyObject*, PyObject*);
PyObject* MCGSM_train(MCGSMObject*, PyObject*, PyObject*);

PyObject* MCGSM_check_gradient(MCGSMObject*, PyObject*, PyObject*);
PyObject* MCGSM_check_performance(MCGSMObject*, PyObject*, PyObject*);

PyObject* MCGSM_sample(MCGSMObject*, PyObject*, PyObject*);
PyObject* MCGSM_sample_posterior(MCGSMObject*, PyObject*, PyObject*);

PyObject* MCGSM_posterior(MCGSMObject*, PyObject*, PyObject*);
PyObject* MCGSM_loglikelihood(MCGSMObject*, PyObject*, PyObject*);

PyObject* MCGSM_parameters(MCGSMObject*, PyObject*, PyObject*);
PyObject* MCGSM_set_parameters(MCGSMObject*, PyObject*, PyObject*);
PyObject* MCGSM_compute_gradient(MCGSMObject*, PyObject*, PyObject*);

PyObject* MCGSM_reduce(MCGSMObject*, PyObject*, PyObject*);
PyObject* MCGSM_setstate(MCGSMObject*, PyObject*, PyObject*);

#endif
