#ifndef TRANSFORMINTERFACE_H
#define TRANSFORMINTERFACE_H

#define PY_ARRAY_UNIQUE_SYMBOL MCM_ARRAY_API
#define NO_IMPORT_ARRAY

#include <Python.h>
#include <arrayobject.h>
#include "lineartransform.h"
#include "whiteningtransform.h"
#include "pyutils.h"

struct TransformObject {
	PyObject_HEAD
	MCM::Transform* transform;
};

struct LinearTransformObject {
	PyObject_HEAD
	MCM::LinearTransform* transform;
};

struct WhiteningTransformObject {
	PyObject_HEAD
	MCM::WhiteningTransform* transform;
};

PyObject* Transform_call(LinearTransformObject*, PyObject*, PyObject*);
PyObject* Transform_inverse(LinearTransformObject*, PyObject*, PyObject*);

PyObject* Transform_new(PyTypeObject*, PyObject*, PyObject*);
void Transform_dealloc(TransformObject*);

int LinearTransform_init(LinearTransformObject*, PyObject*, PyObject*);

PyObject* LinearTransform_A(LinearTransformObject*, PyObject*, void*);
int LinearTransform_set_A(LinearTransformObject*, PyObject*, void*);

PyObject* LinearTransform_reduce(LinearTransformObject*, PyObject*, PyObject*);
PyObject* LinearTransform_setstate(LinearTransformObject*, PyObject*, PyObject*);

int WhiteningTransform_init(WhiteningTransformObject*, PyObject*, PyObject*);

PyObject* WhiteningTransform_reduce(LinearTransformObject*, PyObject*, PyObject*);
PyObject* WhiteningTransform_setstate(LinearTransformObject*, PyObject*, PyObject*);

#endif
