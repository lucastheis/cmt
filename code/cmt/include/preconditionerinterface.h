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

struct PreconditionerObject {
	PyObject_HEAD
	Preconditioner* preconditioner;
};

struct AffinePreconditionerObject {
	PyObject_HEAD
	AffinePreconditioner* preconditioner;
};

struct AffineTransformObject {
	PyObject_HEAD
	AffineTransform* preconditioner;
};

struct WhiteningPreconditionerObject {
	PyObject_HEAD
	WhiteningPreconditioner* preconditioner;
};

struct WhiteningTransformObject {
	PyObject_HEAD
	WhiteningTransform* preconditioner;
};

struct PCAPreconditionerObject {
	PyObject_HEAD
	PCAPreconditioner* preconditioner;
};

extern const char* Preconditioner_doc;
extern const char* Preconditioner_inverse_doc;
extern const char* Preconditioner_logjacobian_doc;
extern const char* AffinePreconditioner_doc;
extern const char* AffineTransform_doc;
extern const char* WhiteningPreconditioner_doc;
extern const char* WhiteningTransform_doc;
extern const char* PCAPreconditioner_doc;

int Preconditioner_init(WhiteningPreconditionerObject*, PyObject*, PyObject*);

PyObject* Preconditioner_call(PreconditionerObject*, PyObject*, PyObject*);
PyObject* Preconditioner_inverse(PreconditionerObject*, PyObject*, PyObject*);

PyObject* Preconditioner_logjacobian(PreconditionerObject*, PyObject*, PyObject*);

PyObject* Preconditioner_new(PyTypeObject*, PyObject*, PyObject*);
void Preconditioner_dealloc(PreconditionerObject*);

PyObject* Preconditioner_dim_in(PreconditionerObject*, PyObject*, void*);
PyObject* Preconditioner_dim_out(PreconditionerObject*, PyObject*, void*);

int AffinePreconditioner_init(AffinePreconditionerObject*, PyObject*, PyObject*);
int AffineTransform_init(AffineTransformObject*, PyObject*, PyObject*);

PyObject* AffinePreconditioner_mean_in(AffinePreconditionerObject*, PyObject*, void*);
PyObject* AffinePreconditioner_mean_out(AffinePreconditionerObject*, PyObject*, void*);

PyObject* AffinePreconditioner_reduce(AffinePreconditionerObject*, PyObject*, PyObject*);
PyObject* AffinePreconditioner_setstate(AffinePreconditionerObject*, PyObject*, PyObject*);

PyObject* AffineTransform_reduce(AffineTransformObject*, PyObject*, PyObject*);

int WhiteningPreconditioner_init(WhiteningPreconditionerObject*, PyObject*, PyObject*);
int WhiteningTransform_init(WhiteningTransformObject*, PyObject*, PyObject*);

int PCAPreconditioner_init(PCAPreconditionerObject*, PyObject*, PyObject*);

PyObject* PCAPreconditioner_reduce(PCAPreconditionerObject*, PyObject*, PyObject*);
PyObject* PCAPreconditioner_setstate(PCAPreconditionerObject*, PyObject*, PyObject*);

#endif
