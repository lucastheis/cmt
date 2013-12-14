#ifndef MIXTUREINTERFACE_H
#define MIXTUREINTERFACE_H

#include <Python.h>
#include <arrayobject.h>

#include <vector>
using std::vector;

#include "cmt/models"
using CMT::Mixture;
using CMT::MoGSM;

struct MixtureObject {
	PyObject_HEAD
	Mixture* mixture;
	bool owner;
	vector<PyTypeObject*> componentTypes;
};

struct MoGSMObject {
	PyObject_HEAD
	MoGSM* mixture;
	bool owner;
	vector<PyTypeObject*> componentTypes;
};

struct MixtureComponentObject {
	PyObject_HEAD
	Mixture::Component* component;
	bool owner;
};

extern PyTypeObject MixtureComponent_type;
extern PyTypeObject GSM_type;

extern const char* Mixture_doc;
extern const char* Mixture_train_doc;
extern const char* Mixture_initialize_doc;
extern const char* MixtureComponent_doc;
extern const char* MixtureComponent_train_doc;
extern const char* MixtureComponent_initialize_doc;
extern const char* MoGSM_doc;

int Mixture_init(MixtureObject*, PyObject*, PyObject*);
int MoGSM_init(MoGSMObject*, PyObject*, PyObject*);

PyObject* Mixture_add_component(MixtureObject*, PyObject*, PyObject*);
PyObject* Mixture_train(MixtureObject*, PyObject*, PyObject*);
PyObject* Mixture_initialize(MixtureObject*, PyObject*, PyObject*);
PyObject* Mixture_subscript(MixtureObject*, PyObject*);
PyObject* Mixture_num_components(MixtureObject*, void*);
PyObject* MoGSM_num_scales(MoGSMObject*, void*);

PyObject* Mixture_priors(MixtureObject*, void*);
int Mixture_set_priors(MixtureObject*, PyObject*, void*);

int MixtureComponent_init(MixtureComponentObject*, PyObject*, PyObject*);
PyObject* MixtureComponent_train(MixtureComponentObject*, PyObject*, PyObject*);
PyObject* MixtureComponent_initialize(MixtureComponentObject*, PyObject*, PyObject*);

PyObject* Mixture_setstate(MixtureObject*, PyObject*);
PyObject* Mixture_reduce(MixtureObject*, PyObject*);
PyObject* MoGSM_reduce(MoGSMObject*, PyObject*);

#endif
