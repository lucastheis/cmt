#ifndef PATCHMODELINTERFACE_H
#define PATCHMODELINTERFACE_H

#define PY_ARRAY_UNIQUE_SYMBOL CMT_ARRAY_API
#define NO_IMPORT_ARRAY

#include <Python.h>
#include <arrayobject.h>

#include "patchmodel.h"
using CMT::PatchModelBase;
using CMT::PatchModel;

struct PatchModelObject {
	PyObject_HEAD
	PatchModelBase* distribution;
	bool owner;
};

extern const char* PatchModel_doc;

PyObject* PatchModel_rows(PatchModelObject*, void*);
PyObject* PatchModel_cols(PatchModelObject*, void*);

PyObject* PatchModel_input_mask(PatchModelObject*, PyObject*);
PyObject* PatchModel_output_mask(PatchModelObject*, PyObject*);

PyObject* PatchModel_subscript(PatchModelObject*, PyObject*);
int PatchModel_ass_subscript(PatchModelObject*, PyObject*, PyObject*);

PyObject* PatchModel_loglikelihood(PatchModelObject*, PyObject*, PyObject*);

#endif
