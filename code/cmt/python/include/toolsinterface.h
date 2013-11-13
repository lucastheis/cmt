#ifndef TOOLSINTERFACE_H
#define TOOLSINTERFACE_H

#define PY_ARRAY_UNIQUE_SYMBOL CMT_ARRAY_API
#define NO_IMPORT_ARRAY

#include <Python.h>
#include <arrayobject.h>
#include "pyutils.h"

extern PyTypeObject Preconditioner_type;
extern PyTypeObject CD_type;
extern PyTypeObject MCGSM_type;

extern const char* random_select_doc;
extern const char* sample_image_doc;
extern const char* sample_image_conditionally_doc;
extern const char* sample_labels_conditionally_doc;
extern const char* sample_video_doc;
extern const char* generate_data_from_image_doc;
extern const char* generate_data_from_video_doc;
extern const char* fill_in_image_doc;
extern const char* extract_windows_doc;
extern const char* sample_spike_train_doc;

PyObject* random_select(PyObject*, PyObject*, PyObject*);
PyObject* generate_data_from_image(PyObject*, PyObject*, PyObject*);
PyObject* generate_data_from_video(PyObject*, PyObject*, PyObject*);
PyObject* sample_image(PyObject*, PyObject*, PyObject*);
PyObject* sample_image_conditionally(PyObject*, PyObject*, PyObject*);
PyObject* sample_labels_conditionally(PyObject*, PyObject*, PyObject*);
PyObject* sample_video(PyObject*, PyObject*, PyObject*);
PyObject* fill_in_image(PyObject*, PyObject*, PyObject*);
PyObject* fill_in_image_map(PyObject*, PyObject*, PyObject*);
PyObject* extract_windows(PyObject*, PyObject*, PyObject*);
PyObject* sample_spike_train(PyObject*, PyObject*, PyObject*);

#endif
