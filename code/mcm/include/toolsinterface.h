#ifndef TOOLSINTERFACE_H
#define TOOLSINTERFACE_H

#define PY_ARRAY_UNIQUE_SYMBOL MCM_ARRAY_API
#define NO_IMPORT_ARRAY

#include <Python.h>
#include <arrayobject.h>
#include "pyutils.h"
#include "conditionaldistribution.h"

struct CDObject {
	PyObject_HEAD
	ConditionalDistribution* cd;
};

extern const char* random_select_doc;
extern const char* sample_video_doc;
extern const char* generate_data_from_video_doc;

PyObject* random_select(PyObject*, PyObject*, PyObject*);
PyObject* generate_data_from_image(PyObject*, PyObject*, PyObject*);
PyObject* generate_data_from_video(PyObject*, PyObject*, PyObject*);
PyObject* sample_image(PyObject*, PyObject*, PyObject*);
PyObject* sample_video(PyObject*, PyObject*, PyObject*);

#endif
