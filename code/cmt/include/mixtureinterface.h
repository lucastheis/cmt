#ifndef MIXTUREINTERFACE_H
#define MIXTUREINTERFACE_H

#include <Python.h>
#include <arrayobject.h>

#include <vector>
using std::vector;

#include "mixture.h"
using CMT::Mixture;

struct MixtureObject {
	PyObject_HEAD
	Mixture* mixture;
	bool owner;
};

struct MixtureComponentObject {
	PyObject_HEAD
	Mixture::Component* component;
	bool owner;
};

extern PyTypeObject MixtureComponent_type;

int Mixture_init(MixtureObject*, PyObject*, PyObject*);

PyObject* Mixture_add_component(MixtureObject*, PyObject*, PyObject*);
PyObject* Mixture_train(MixtureObject*, PyObject*, PyObject*);
PyObject* Mixture_subscript(MixtureObject*, PyObject*);

PyObject* Mixture_priors(MixtureObject*, void*);
int Mixture_set_priors(MixtureObject*, PyObject*, void*);

int MixtureComponent_init(MixtureComponentObject*, PyObject*, PyObject*);
PyObject* MixtureComponent_train(MixtureComponentObject*, PyObject*, PyObject*);

PyObject* Mixture_reduce(MixtureObject*, PyObject*);
PyObject* Mixture_setstate(MixtureObject*, PyObject*);

#endif
