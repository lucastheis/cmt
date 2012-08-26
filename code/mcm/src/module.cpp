#define PY_ARRAY_UNIQUE_SYMBOL MCM_ARRAY_API

#include <Python.h>
#include <arrayobject.h>
#include <structmember.h>
#include <stdlib.h>
#include <time.h>
#include "mcgsminterface.h"
#include "Eigen/Core"

#pragma GCC diagnostic ignored "-Wwrite-strings"

static PyGetSetDef MCGSM_getset[] = {
	{"dim_in", (getter)MCGSM_dim_in, 0, 0},
	{"dim_out", (getter)MCGSM_dim_out, 0, 0},
	{"num_components", (getter)MCGSM_num_components, 0, 0},
	{"num_scales", (getter)MCGSM_num_scales, 0, 0},
	{"num_features", (getter)MCGSM_num_features, 0, 0},
	{"priors", (getter)MCGSM_priors, (setter)MCGSM_set_priors, 0},
	{"scales", (getter)MCGSM_scales, (setter)MCGSM_set_scales, 0},
	{"weights", (getter)MCGSM_weights, (setter)MCGSM_set_weights, 0},
	{"features", (getter)MCGSM_features, (setter)MCGSM_set_features, 0},
	{"cholesky_factors", (getter)MCGSM_cholesky_factors, (setter)MCGSM_set_cholesky_factors, 0},
	{"predictors", (getter)MCGSM_predictors, (setter)MCGSM_set_predictors, 0},
	{0}
};



static PyMethodDef MCGSM_methods[] = {
	{"normalize", (PyCFunction)MCGSM_normalize, METH_NOARGS, 0},
	{"train", (PyCFunction)MCGSM_train, METH_VARARGS|METH_KEYWORDS, 0},
	{"posterior", (PyCFunction)MCGSM_posterior, METH_VARARGS|METH_KEYWORDS, 0},
	{"sample", (PyCFunction)MCGSM_sample, METH_VARARGS|METH_KEYWORDS, 0},
	{"sample_posterior", (PyCFunction)MCGSM_sample_posterior, METH_VARARGS|METH_KEYWORDS, 0},
	{"loglikelihood", (PyCFunction)MCGSM_loglikelihood, METH_VARARGS|METH_KEYWORDS, 0},
	{"__reduce__", (PyCFunction)MCGSM_reduce, METH_NOARGS, 0},
	{"__setstate__", (PyCFunction)MCGSM_setstate, METH_VARARGS, 0},
	{0}
};



PyTypeObject MCGSM_type = {
	PyObject_HEAD_INIT(0)
	0,                         /*ob_size*/
	"mcm.MCGSM",               /*tp_name*/
	sizeof(MCGSMObject),       /*tp_basicsize*/
	0,                         /*tp_itemsize*/
	(destructor)MCGSM_dealloc, /*tp_dealloc*/
	0,                         /*tp_print*/
	0,                         /*tp_getattr*/
	0,                         /*tp_setattr*/
	0,                         /*tp_compare*/
	0,                         /*tp_repr*/
	0,                         /*tp_as_number*/
	0,                         /*tp_as_sequence*/
	0,                         /*tp_as_mapping*/
	0,                         /*tp_hash */
	0,                         /*tp_call*/
	0,                         /*tp_str*/
	0,                         /*tp_getattro*/
	0,                         /*tp_setattro*/
	0,                         /*tp_as_buffer*/
	Py_TPFLAGS_DEFAULT,        /*tp_flags*/
	0,                         /*tp_doc*/
	0,                         /*tp_traverse*/
	0,                         /*tp_clear*/
	0,                         /*tp_richcompare*/
	0,                         /*tp_weaklistoffset*/
	0,                         /*tp_iter*/
	0,                         /*tp_iternext*/
	MCGSM_methods,             /*tp_methods*/
	0,                         /*tp_members*/
	MCGSM_getset,              /*tp_getset*/
	0,                         /*tp_base*/
	0,                         /*tp_dict*/
	0,                         /*tp_descr_get*/
	0,                         /*tp_descr_set*/
	0,                         /*tp_dictoffset*/
	(initproc)MCGSM_init,      /*tp_init*/
	0,                         /*tp_alloc*/
	MCGSM_new,                 /*tp_new*/
};



PyMODINIT_FUNC initmcm() {
	// set random seed
	timeval time;
	gettimeofday(&time, 0);
	srand(time.tv_usec * time.tv_sec);

	// initialize NumPy
	import_array();

	// create module object
	PyObject* module = Py_InitModule("mcm", 0);

	// initialize types
	if(PyType_Ready(&MCGSM_type) < 0)
		return;

	// initialize Eigen
	Eigen::initParallel();

	// add types to module
	Py_INCREF(&MCGSM_type);
	PyModule_AddObject(module, "MCGSM", reinterpret_cast<PyObject*>(&MCGSM_type));
}
