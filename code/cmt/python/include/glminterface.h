#ifndef GLMINTERFACE_H
#define GLMINTERFACE_H

#define PY_ARRAY_UNIQUE_SYMBOL CMT_ARRAY_API
#define NO_IMPORT_ARRAY

#include <Python.h>
#include <arrayobject.h>
#include "pyutils.h"
#include "trainableinterface.h"

#include "cmt/models"
using CMT::GLM;
using CMT::LogisticFunction;
using CMT::Bernoulli;

struct NonlinearityObject {
	PyObject_HEAD
	GLM::Nonlinearity* nonlinearity;
	bool owner;
};

struct LogisticFunctionObject {
	PyObject_HEAD
	LogisticFunction* nonlinearity;
	bool owner;
};

struct UnivariateDistributionObject {
	PyObject_HEAD
	GLM::UnivariateDistribution* distribution;
	bool owner;
};

struct BernoulliObject {
	PyObject_HEAD
	Bernoulli* distribution;
	bool owner;
};

struct GLMObject {
	PyObject_HEAD
	GLM* glm;
	bool owner;
	NonlinearityObject* nonlinearity;
	UnivariateDistributionObject* distribution;
};

extern PyTypeObject Nonlinearity_type;
extern PyTypeObject UnivariateDistribution_type;
extern PyTypeObject GLM_type;

extern const char* Nonlinearity_doc;
extern const char* LogisticFunction_doc;
extern const char* LogisticFunction_reduce_doc;

extern const char* UnivariateDistribution_doc;
extern const char* Bernoulli_doc;
extern const char* Bernoulli_reduce_doc;

extern const char* GLM_doc;
extern const char* GLM_reduce_doc;
extern const char* GLM_setstate_doc;

PyObject* Nonlinearity_new(PyTypeObject* type, PyObject* args, PyObject* kwds);
int Nonlinearity_init(NonlinearityObject*, PyObject*, PyObject*);
void Nonlinearity_dealloc(NonlinearityObject*);
PyObject* Nonlinearity_call(NonlinearityObject*, PyObject*, PyObject*);

int LogisticFunction_init(LogisticFunctionObject*, PyObject*, PyObject*);
PyObject* LogisticFunction_reduce(LogisticFunctionObject*, PyObject*);

int UnivariateDistribution_init(UnivariateDistributionObject*, PyObject*, PyObject*);

int Bernoulli_init(BernoulliObject*, PyObject*, PyObject*);
PyObject* Bernoulli_reduce(BernoulliObject*, PyObject*);

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

PyObject* GLM_parameter_gradient(GLMObject*, PyObject*, PyObject*);
PyObject* GLM_check_gradient(GLMObject*, PyObject*, PyObject*);

PyObject* GLM_reduce(GLMObject*, PyObject*);
PyObject* GLM_setstate(GLMObject*, PyObject*);

Trainable::Parameters* PyObject_ToGLMParameters(PyObject* parameters);

#endif
