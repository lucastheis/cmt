#ifndef TRANSFORMINTERFACE_H
#define TRANSFORMINTERFACE_H

#define PY_ARRAY_UNIQUE_SYMBOL MCM_ARRAY_API
#define NO_IMPORT_ARRAY

#include <Python.h>
#include <arrayobject.h>
#include "pyutils.h"

#include "lineartransform.h"
using MCM::Transform;
using MCM::LinearTransform;

#include "affinetransform.h"
using MCM::AffineTransform;

#include "whiteningtransform.h"
using MCM::WhiteningTransform;

#include "pcatransform.h"
using MCM::PCATransform;

struct TransformObject {
	PyObject_HEAD
	Transform* transform;
};

struct AffineTransformObject {
	PyObject_HEAD
	AffineTransform* transform;
};

struct LinearTransformObject {
	PyObject_HEAD
	LinearTransform* transform;
};

struct WhiteningTransformObject {
	PyObject_HEAD
	WhiteningTransform* transform;
};

struct PCATransformObject {
	PyObject_HEAD
	PCATransform* transform;
};

PyObject* Transform_call(LinearTransformObject*, PyObject*, PyObject*);
PyObject* Transform_inverse(LinearTransformObject*, PyObject*, PyObject*);

PyObject* Transform_new(PyTypeObject*, PyObject*, PyObject*);
void Transform_dealloc(TransformObject*);

PyObject* Transform_dim_in(TransformObject*, PyObject*, void*);
PyObject* Transform_dim_out(TransformObject*, PyObject*, void*);

int AffineTransform_init(AffineTransformObject*, PyObject*, PyObject*);

PyObject* AffineTransform_A(AffineTransformObject*, PyObject*, void*);
int AffineTransform_set_A(AffineTransformObject*, PyObject*, void*);

PyObject* AffineTransform_b(AffineTransformObject*, PyObject*, void*);
int AffineTransform_set_b(AffineTransformObject*, PyObject*, void*);

PyObject* AffineTransform_reduce(AffineTransformObject*, PyObject*, PyObject*);
PyObject* AffineTransform_setstate(AffineTransformObject*, PyObject*, PyObject*);

int LinearTransform_init(LinearTransformObject*, PyObject*, PyObject*);

PyObject* LinearTransform_reduce(AffineTransformObject*, PyObject*, PyObject*);
PyObject* LinearTransform_setstate(AffineTransformObject*, PyObject*, PyObject*);

int WhiteningTransform_init(WhiteningTransformObject*, PyObject*, PyObject*);

PyObject* WhiteningTransform_reduce(AffineTransformObject*, PyObject*, PyObject*);
PyObject* WhiteningTransform_setstate(AffineTransformObject*, PyObject*, PyObject*);

int PCATransform_init(PCATransformObject*, PyObject*, PyObject*);

PyObject* PCATransform_eigenvalues(PCATransformObject*, PyObject*, PyObject*);

#endif
