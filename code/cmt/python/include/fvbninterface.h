#ifndef FVBNINTERFACE_H
#define FVBNINTERFACE_H

#define PY_ARRAY_UNIQUE_SYMBOL CMT_ARRAY_API
#define NO_IMPORT_ARRAY

#include <Python.h>
#include <arrayobject.h>

#include "cmt/models"
using CMT::GLM;
using CMT::LogisticFunction;
using CMT::Bernoulli;
using CMT::PatchModel;

#include "cmt/transforms"
using CMT::PCATransform;

struct FVBNObject {
	PyObject_HEAD
	PatchModel<GLM, PCATransform>* fvbn;
	bool owner;
	PyTypeObject* distributionType;
	PyTypeObject* nonlinearityType;
};

extern const char* FVBN_doc;
extern const char* FVBN_initialize_doc;
extern const char* FVBN_train_doc;
extern const char* FVBN_reduce_doc;
extern const char* FVBN_setstate_doc;

extern PyTypeObject GLM_type;
extern PyTypeObject LogisticFunction_type;
extern PyTypeObject Bernoulli_type;
extern PyTypeObject PCATransform_type;

int FVBN_init(FVBNObject*, PyObject*, PyObject*);

PyObject* FVBN_subscript(FVBNObject*, PyObject*);
int FVBN_ass_subscript(FVBNObject*, PyObject*, PyObject*);

PyObject* FVBN_preconditioner(FVBNObject*, PyObject*);
PyObject* FVBN_preconditioners(FVBNObject*, void*);
int FVBN_set_preconditioners(FVBNObject*, PyObject*, void*);

PyObject* FVBN_initialize(FVBNObject*, PyObject*, PyObject*);
PyObject* FVBN_train(FVBNObject*, PyObject*, PyObject*);

PyObject* FVBN_reduce(FVBNObject*, PyObject*);
PyObject* FVBN_setstate(FVBNObject*, PyObject*);

#endif
