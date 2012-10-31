#ifndef TOOLSINTERFACE_H
#define TOOLSINTERFACE_H

#define PY_ARRAY_UNIQUE_SYMBOL MCM_ARRAY_API
#define NO_IMPORT_ARRAY

#include <Python.h>
#include <arrayobject.h>
#include "tools.h"
#include "pyutils.h"

struct CDObject {
	PyObject_HEAD
	ConditionalDistribution* cd;
};

PyObject* sample_image(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* shuffle(PyObject* self, PyObject* args, PyObject* kwds);

#endif
