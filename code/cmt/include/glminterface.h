#ifndef GLMINTERFACE_H
#define GLMINTERFACE_H

#include <Python.h>
#include <arrayobject.h>
#include "pyutils.h"

#include "glm.h"
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
	Bernoulli* distribution;
	bool owner;
};

struct BernoulliObject {
	PyObject_HEAD
	GLM::UnivariateDistribution* distribution;
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

extern const char* Nonlinearity_doc;
extern const char* LogisticFunction_doc;

extern const char* UnivariateDistribution_doc;
extern const char* Bernoulli_doc;

extern const char* GLM_doc;

PyObject* Nonlinearity_new(PyTypeObject* type, PyObject* args, PyObject* kwds);
int Nonlinearity_init(NonlinearityObject*, PyObject*, PyObject*);
void Nonlinearity_dealloc(NonlinearityObject*);
int LogisticFunction_init(LogisticFunctionObject*, PyObject*, PyObject*);

int UnivariateDistribution_init(UnivariateDistributionObject*, PyObject*, PyObject*);
int Bernoulli_init(BernoulliObject*, PyObject*, PyObject*);

int GLM_init(GLMObject*, PyObject*, PyObject*);
void GLM_dealloc(GLMObject*);

PyObject* GLM_weights(GLMObject*, void*);
int GLM_set_weights(GLMObject*, PyObject*, void*);

//PyObject* GLM_reduce(GLMObject*, PyObject*);
//PyObject* GLM_setstate(GLMObject*, PyObject*);

#endif
