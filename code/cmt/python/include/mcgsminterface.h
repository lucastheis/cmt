#ifndef MCGSMINTERFACE_H
#define MCGSMINTERFACE_H

#define PY_ARRAY_UNIQUE_SYMBOL CMT_ARRAY_API
#define NO_IMPORT_ARRAY

#include <Python.h>
#include <arrayobject.h>
#include "pyutils.h"

#include "cmt/models"
using CMT::MCGSM;
using CMT::PatchModel;

#include "cmt/transforms"
using CMT::PCAPreconditioner;

struct MCGSMObject {
	PyObject_HEAD
	MCGSM* mcgsm;
	bool owner;
};

struct PatchMCGSMObject {
	PyObject_HEAD
	PatchModel<MCGSM, PCAPreconditioner>* patchMCGSM;
	bool owner;
};

extern PyTypeObject MCGSM_type;
extern PyTypeObject PCAPreconditioner_type;

extern const char* MCGSM_doc;
extern const char* MCGSM_train_doc;
extern const char* MCGSM_loglikelihood_doc;
extern const char* MCGSM_sample_doc;
extern const char* MCGSM_sample_prior_doc;
extern const char* MCGSM_sample_posterior_doc;
extern const char* MCGSM_prior_doc;
extern const char* MCGSM_posterior_doc;
extern const char* MCGSM_reduce_doc;
extern const char* MCGSM_setstate_doc;

extern const char* PatchMCGSM_doc;
extern const char* PatchMCGSM_initialize_doc;
extern const char* PatchMCGSM_train_doc;
extern const char* PatchMCGSM_reduce_doc;
extern const char* PatchMCGSM_setstate_doc;

int MCGSM_init(MCGSMObject*, PyObject*, PyObject*);

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

PyObject* MCGSM_linear_features(MCGSMObject*, PyObject*, void*);
int MCGSM_set_linear_features(MCGSMObject*, PyObject*, void*);

PyObject* MCGSM_means(MCGSMObject*, PyObject*, void*);
int MCGSM_set_means(MCGSMObject*, PyObject*, void*);

PyObject* MCGSM_train(MCGSMObject*, PyObject*, PyObject*);

PyObject* MCGSM_check_gradient(MCGSMObject*, PyObject*, PyObject*);
PyObject* MCGSM_check_performance(MCGSMObject*, PyObject*, PyObject*);

PyObject* MCGSM_loglikelihood(MCGSMObject*, PyObject*, PyObject*);

PyObject* MCGSM_sample(MCGSMObject*, PyObject*, PyObject*);
PyObject* MCGSM_sample_prior(MCGSMObject*, PyObject*, PyObject*);
PyObject* MCGSM_sample_posterior(MCGSMObject*, PyObject*, PyObject*);
PyObject* MCGSM_prior(MCGSMObject*, PyObject*, PyObject*);
PyObject* MCGSM_posterior(MCGSMObject*, PyObject*, PyObject*);

PyObject* MCGSM_parameters(MCGSMObject*, PyObject*, PyObject*);
PyObject* MCGSM_set_parameters(MCGSMObject*, PyObject*, PyObject*);
PyObject* MCGSM_parameter_gradient(MCGSMObject*, PyObject*, PyObject*);

PyObject* MCGSM_compute_data_gradient(MCGSMObject*, PyObject*, PyObject*);

PyObject* MCGSM_reduce(MCGSMObject*, PyObject*, PyObject*);
PyObject* MCGSM_setstate(MCGSMObject*, PyObject*, PyObject*);

int PatchMCGSM_init(PatchMCGSMObject*, PyObject*, PyObject*);

PyObject* PatchMCGSM_subscript(PatchMCGSMObject*, PyObject* key);
int PatchMCGSM_ass_subscript(PatchMCGSMObject*, PyObject*, PyObject*);

PyObject* PatchMCGSM_preconditioner(PatchMCGSMObject*, PyObject*);
PyObject* PatchMCGSM_preconditioners(PatchMCGSMObject*, void*);
int PatchMCGSM_set_preconditioners(PatchMCGSMObject*, PyObject*, void*);

PyObject* PatchMCGSM_initialize(PatchMCGSMObject*, PyObject*, PyObject*);
PyObject* PatchMCGSM_train(PatchMCGSMObject*, PyObject*, PyObject*);

PyObject* PatchMCGSM_reduce(PatchMCGSMObject*, PyObject*);
PyObject* PatchMCGSM_setstate(PatchMCGSMObject*, PyObject*);

#endif
