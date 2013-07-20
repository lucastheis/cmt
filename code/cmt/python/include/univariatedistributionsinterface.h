#ifndef UNIVARIATEDISTRIBUIONSINTERFACE_H
#define UNIVARIATEDISTRIBUIONSINTERFACE_H

#define PY_ARRAY_UNIQUE_SYMBOL CMT_ARRAY_API
#define NO_IMPORT_ARRAY

#include <Python.h>
#include <arrayobject.h>
#include "pyutils.h"

#include "cmt/models"
using CMT::UnivariateDistribution;
using CMT::Bernoulli;

struct UnivariateDistributionObject {
	PyObject_HEAD
	UnivariateDistribution* distribution;
	bool owner;
};

struct BernoulliObject {
	PyObject_HEAD
	Bernoulli* distribution;
	bool owner;
};

extern const char* UnivariateDistribution_doc;
extern const char* Bernoulli_doc;
extern const char* Bernoulli_reduce_doc;

int UnivariateDistribution_init(UnivariateDistributionObject*, PyObject*, PyObject*);

int Bernoulli_init(BernoulliObject*, PyObject*, PyObject*);
PyObject* Bernoulli_reduce(BernoulliObject*, PyObject*);

#endif
