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
using CMT::Poisson;
using CMT::Binomial;

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

struct PoissonObject {
	PyObject_HEAD
	Poisson* distribution;
	bool owner;
};

struct BinomialObject {
	PyObject_HEAD
	Binomial* distribution;
	bool owner;
};

extern const char* UnivariateDistribution_doc;
extern const char* Bernoulli_doc;
extern const char* Bernoulli_reduce_doc;
extern const char* Poisson_doc;
extern const char* Poisson_reduce_doc;
extern const char* Binomial_doc;
extern const char* Binomial_reduce_doc;

int UnivariateDistribution_init(UnivariateDistributionObject*, PyObject*, PyObject*);

int Bernoulli_init(BernoulliObject*, PyObject*, PyObject*);
PyObject* Bernoulli_reduce(BernoulliObject*, PyObject*);

int Poisson_init(PoissonObject*, PyObject*, PyObject*);
PyObject* Poisson_reduce(PoissonObject*, PyObject*);

int Binomial_init(BinomialObject*, PyObject*, PyObject*);
PyObject* Binomial_reduce(BinomialObject*, PyObject*);

#endif
