#ifndef PRECONDITIONERINTERFACE_H
#define PRECONDITIONERINTERFACE_H

#define PY_ARRAY_UNIQUE_SYMBOL MCM_ARRAY_API
#define NO_IMPORT_ARRAY

#include <Python.h>
#include <arrayobject.h>
#include "pyutils.h"

#include "whiteningpreconditioner.h"
using MCM::Preconditioner;
using MCM::WhiteningPreconditioner;

struct PreconditionerObject {
	PyObject_HEAD
	Preconditioner* preconditioner;
};

struct WhiteningPreconditionerObject {
	PyObject_HEAD
	WhiteningPreconditioner* preconditioner;
};

PyObject* Preconditioner_call(PreconditionerObject*, PyObject*, PyObject*);
PyObject* Preconditioner_inverse(PreconditionerObject*, PyObject*, PyObject*);

PyObject* Preconditioner_new(PyTypeObject*, PyObject*, PyObject*);
void Preconditioner_dealloc(PreconditionerObject*);

PyObject* Preconditioner_dim_in(PreconditionerObject*, PyObject*, void*);
PyObject* Preconditioner_dim_out(PreconditionerObject*, PyObject*, void*);

int WhiteningPreconditioner_init(WhiteningPreconditionerObject*, PyObject*, PyObject*);

PyObject* WhiteningPreconditioner_reduce(WhiteningPreconditionerObject*, PyObject*, PyObject*);
PyObject* WhiteningPreconditioner_setstate(WhiteningPreconditionerObject*, PyObject*, PyObject*);

#endif
