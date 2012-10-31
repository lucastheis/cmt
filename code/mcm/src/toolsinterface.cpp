#include "toolsinterface.h"
#include "mcgsminterface.h"
#include "exception.h"

PyObject* sample_image(PyObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"img", "model", "xmask", "ymask", 0};

	PyObject* img;
	PyObject* model;
	PyObject* xmask;
	PyObject* ymask;

	if(!PyArg_ParseTupleAndKeywords(args, kwds, "OOOO", const_cast<char**>(kwlist),
		&img, &model, &xmask, &ymask))
		return 0;

	// make sure data is stored in NumPy array
	img = PyArray_FROM_OTF(img, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
	xmask = PyArray_FROM_OTF(xmask, NPY_BOOL, NPY_F_CONTIGUOUS | NPY_ALIGNED);
	ymask = PyArray_FROM_OTF(ymask, NPY_BOOL, NPY_F_CONTIGUOUS | NPY_ALIGNED);

	const ConditionalDistribution& cd = *reinterpret_cast<CDObject*>(model)->cd;

	if(!img) {
		PyErr_SetString(PyExc_TypeError, "The initial image has to be given as an array.");
		return 0;
	}

	if(!xmask || !ymask) {
		Py_DECREF(img);
		Py_XDECREF(xmask);
		Py_XDECREF(ymask);
		PyErr_SetString(PyExc_TypeError, "Masks have to be given as Boolean arrays.");
		return 0;
	}

	try {
		PyObject* imgSample = PyArray_FromMatrixXd(
			sampleImage(
				PyArray_ToMatrixXd(img),
				cd,
				PyArray_ToMatrixXb(xmask),
				PyArray_ToMatrixXb(ymask)));
		Py_DECREF(img);
		Py_DECREF(xmask);
		Py_DECREF(ymask);
		return imgSample;
	} catch(Exception exception) {
		Py_DECREF(img);
		Py_DECREF(xmask);
		Py_DECREF(ymask);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	return 0;
}
