#ifndef PRECONDITIONERINTERFACE_H
#define PRECONDITIONERINTERFACE_H

#define PY_ARRAY_UNIQUE_SYMBOL CMT_ARRAY_API
#define NO_IMPORT_ARRAY

#include <Python.h>
#include <arrayobject.h>
#include "pyutils.h"

#include "affinepreconditioner.h"
using CMT::Preconditioner;
using CMT::AffinePreconditioner;

#include "affinetransform.h"
using CMT::AffineTransform;

#include "whiteningpreconditioner.h"
using CMT::WhiteningPreconditioner;

#include "whiteningtransform.h"
using CMT::WhiteningTransform;

#include "pcapreconditioner.h"
using CMT::PCAPreconditioner;

#include "pcatransform.h"
using CMT::PCATransform;

struct PreconditionerObject {
	PyObject_HEAD
	Preconditioner* preconditioner;
	bool owner;
};

struct AffinePreconditionerObject {
	PyObject_HEAD
	AffinePreconditioner* preconditioner;
	bool owner;
};

struct AffineTransformObject {
	PyObject_HEAD
	AffineTransform* preconditioner;
	bool owner;
};

struct WhiteningPreconditionerObject {
	PyObject_HEAD
	WhiteningPreconditioner* preconditioner;
	bool owner;
};

struct WhiteningTransformObject {
	PyObject_HEAD
	WhiteningTransform* preconditioner;
	bool owner;
};

struct PCAPreconditionerObject {
	PyObject_HEAD
	PCAPreconditioner* preconditioner;
	bool owner;
};

struct PCATransformObject {
	PyObject_HEAD
	PCATransform* preconditioner;
	bool owner;
};

extern const char* Preconditioner_doc;
extern const char* Preconditioner_inverse_doc;
extern const char* Preconditioner_logjacobian_doc;
extern const char* AffinePreconditioner_doc;
extern const char* AffineTransform_doc;
extern const char* WhiteningPreconditioner_doc;
extern const char* WhiteningTransform_doc;
extern const char* PCAPreconditioner_doc;
extern const char* PCATransform_doc;

int Preconditioner_init(WhiteningPreconditionerObject*, PyObject*, PyObject*);

PyObject* Preconditioner_call(PreconditionerObject*, PyObject*, PyObject*);
PyObject* Preconditioner_inverse(PreconditionerObject*, PyObject*, PyObject*);

PyObject* Preconditioner_logjacobian(PreconditionerObject*, PyObject*, PyObject*);

PyObject* Preconditioner_new(PyTypeObject*, PyObject*, PyObject*);
void Preconditioner_dealloc(PreconditionerObject*);

PyObject* Preconditioner_dim_in(PreconditionerObject*, void*);
PyObject* Preconditioner_dim_out(PreconditionerObject*, void*);

int AffinePreconditioner_init(AffinePreconditionerObject*, PyObject*, PyObject*);
int AffineTransform_init(AffineTransformObject*, PyObject*, PyObject*);

PyObject* AffinePreconditioner_mean_in(AffinePreconditionerObject*, void*);
PyObject* AffinePreconditioner_mean_out(AffinePreconditionerObject*, void*);
PyObject* AffinePreconditioner_pre_in(AffinePreconditionerObject*, void*);
PyObject* AffinePreconditioner_pre_out(AffinePreconditionerObject*, void*);
PyObject* AffinePreconditioner_predictor(AffinePreconditionerObject*, void*);

PyObject* AffinePreconditioner_reduce(AffinePreconditionerObject*, PyObject*);
PyObject* AffinePreconditioner_setstate(AffinePreconditionerObject*, PyObject*);

PyObject* AffineTransform_reduce(AffineTransformObject*, PyObject*);

int WhiteningPreconditioner_init(WhiteningPreconditionerObject*, PyObject*, PyObject*);
int WhiteningTransform_init(WhiteningTransformObject*, PyObject*, PyObject*);

int PCAPreconditioner_init(PCAPreconditionerObject*, PyObject*, PyObject*);
int PCATransform_init(PCATransformObject*, PyObject*, PyObject*);

PyObject* PCAPreconditioner_eigenvalues(PCAPreconditionerObject*, void*);
PyObject* PCATransform_eigenvalues(PCATransformObject*, void*);

PyObject* PCAPreconditioner_reduce(PCAPreconditionerObject*, PyObject*);
PyObject* PCATransform_reduce(PCATransformObject*, PyObject*);

#endif
