#ifndef STMINTERFACE_H
#define STMINTERFACE_H

#define PY_ARRAY_UNIQUE_SYMBOL CMT_ARRAY_API
#define NO_IMPORT_ARRAY

#include <Python.h>
#include <arrayobject.h>
#include "pyutils.h"
#include "nonlinearitiesinterface.h"
#include "univariatedistributionsinterface.h"

#include "cmt/models"
using CMT::STM;
using CMT::PatchModel;

#include "cmt/transforms"
using CMT::PCATransform;

struct STMObject {
	PyObject_HEAD
	STM* stm;
	bool owner;
	NonlinearityObject* nonlinearity;
	UnivariateDistributionObject* distribution;
};

extern PyTypeObject STM_type;
extern PyTypeObject Nonlinearity_type;
extern PyTypeObject LogisticFunction_type;
extern PyTypeObject UnivariateDistribution_type;
extern PyTypeObject Bernoulli_type;

extern const char* STM_doc;
extern const char* STM_nonlinear_responses_doc;
extern const char* STM_linear_response_doc;
extern const char* STM_train_doc;
extern const char* STM_sample_posterior_doc;
extern const char* STM_reduce_doc;
extern const char* STM_setstate_doc;

int STM_init(STMObject*, PyObject*, PyObject*);

PyObject* STM_dim_in_nonlinear(STMObject*, void*);
PyObject* STM_dim_in_linear(STMObject*, void*);
PyObject* STM_num_components(STMObject*, void*);
PyObject* STM_num_features(STMObject*, void*);

PyObject* STM_sharpness(STMObject*, void*);
int STM_set_sharpness(STMObject*, PyObject*, void*);

PyObject* STM_biases(STMObject*, void*);
int STM_set_biases(STMObject*, PyObject*, void*);

PyObject* STM_weights(STMObject*, void*);
int STM_set_weights(STMObject*, PyObject*, void*);

PyObject* STM_features(STMObject*, void*);
int STM_set_features(STMObject*, PyObject*, void*);

PyObject* STM_predictors(STMObject*, void*);
int STM_set_predictors(STMObject*, PyObject*, void*);

PyObject* STM_linear_predictor(STMObject*, void*);
int STM_set_linear_predictor(STMObject*, PyObject*, void*);

PyObject* STM_nonlinearity(STMObject*, void*);
int STM_set_nonlinearity(STMObject*, PyObject*, void*);

PyObject* STM_distribution(STMObject*, void*);
int STM_set_distribution(STMObject*, PyObject*, void*);

PyObject* STM_linear_response(STMObject*, PyObject*, PyObject*);
PyObject* STM_nonlinear_responses(STMObject*, PyObject*, PyObject*);

PyObject* STM_train(STMObject*, PyObject*, PyObject*);

PyObject* STM_parameters(STMObject*, PyObject*, PyObject*);
PyObject* STM_set_parameters(STMObject*, PyObject*, PyObject*);
PyObject* STM_parameter_gradient(STMObject*, PyObject*, PyObject*);
PyObject* STM_fisher_information(STMObject*, PyObject*, PyObject*);
PyObject* STM_check_gradient(STMObject*, PyObject*, PyObject*);
PyObject* STM_check_performance(STMObject* self, PyObject* args, PyObject* kwds);

PyObject* STM_reduce(STMObject*, PyObject*);
PyObject* STM_setstate(STMObject*, PyObject*);

#endif
