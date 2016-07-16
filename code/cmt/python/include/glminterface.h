#ifndef GLMINTERFACE_H
#define GLMINTERFACE_H

#define PY_ARRAY_UNIQUE_SYMBOL CMT_ARRAY_API
#define NO_IMPORT_ARRAY

#include <Python.h>
#include <arrayobject.h>
#include "pyutils.h"
#include "trainableinterface.h"
#include "nonlinearitiesinterface.h"
#include "univariatedistributionsinterface.h"

#include "cmt/models"
using CMT::GLM;

struct GLMObject {
	PyObject_HEAD
	GLM* glm;
	bool owner;
	NonlinearityObject* nonlinearity;
	UnivariateDistributionObject* distribution;
};

extern PyTypeObject GLM_type;
extern PyTypeObject Nonlinearity_type;
extern PyTypeObject LogisticFunction_type;
extern PyTypeObject UnivariateDistribution_type;
extern PyTypeObject Bernoulli_type;

extern const char* GLM_doc;
extern const char* GLM_train_doc;
extern const char* GLM_reduce_doc;
extern const char* GLM_setstate_doc;

int GLM_init(GLMObject*, PyObject*, PyObject*);
void GLM_dealloc(GLMObject*);

PyObject* GLM_weights(GLMObject*, void*);
int GLM_set_weights(GLMObject*, PyObject*, void*);

PyObject* GLM_bias(GLMObject*, void*);
int GLM_set_bias(GLMObject*, PyObject*, void*);

PyObject* GLM_nonlinearity(GLMObject*, void*);
int GLM_set_nonlinearity(GLMObject*, PyObject*, void*);

PyObject* GLM_distribution(GLMObject*, void*);
int GLM_set_distribution(GLMObject*, PyObject*, void*);

PyObject* GLM_train(GLMObject*, PyObject*, PyObject*);

PyObject* GLM_parameters(GLMObject*, PyObject*, PyObject*);
PyObject* GLM_set_parameters(GLMObject*, PyObject*, PyObject*);
PyObject* GLM_parameter_gradient(GLMObject*, PyObject*, PyObject*);
PyObject* GLM_fisher_information(GLMObject*, PyObject*, PyObject*);
PyObject* GLM_check_gradient(GLMObject*, PyObject*, PyObject*);
PyObject* GLM_check_performance(GLMObject* self, PyObject* args, PyObject* kwds);

PyObject* GLM_reduce(GLMObject*, PyObject*);
PyObject* GLM_setstate(GLMObject*, PyObject*);

Trainable::Parameters* PyObject_ToGLMParameters(PyObject* parameters);

#endif
