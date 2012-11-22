#include "toolsinterface.h"
#include "mcgsminterface.h"
#include "transforminterface.h"
#include "exception.h"
#include "utils.h"
#include "tools.h"

#include <set>
using std::set;

#include <iostream>

PyObject* random_select(PyObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"k", "n", 0};

	int n;
	int k;

	if(!PyArg_ParseTupleAndKeywords(args, kwds, "ii", const_cast<char**>(kwlist), &k, &n))
		return 0;

	try {
		set<int> indices = randomSelect(k, n);

		PyObject* list = PyList_New(indices.size());
		
		int i = 0;

		for(set<int>::iterator iter = indices.begin(); iter != indices.end(); ++iter, ++i)
			PyList_SetItem(list, i, PyInt_FromLong(*iter));

		return list;
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}
	
	return 0;
}



PyObject* generate_data_from_image(PyObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"img", "xmask", "ymask", "num_samples", 0};

	PyObject* img;
	PyObject* xmask;
	PyObject* ymask;
	int num_samples;

	if(!PyArg_ParseTupleAndKeywords(args, kwds, "OOOi", const_cast<char**>(kwlist),
		&img, &xmask, &ymask, &num_samples))
		return 0;

	// make sure data is stored in NumPy array
	img = PyArray_FROM_OTF(img, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
	xmask = PyArray_FROM_OTF(xmask, NPY_BOOL, NPY_F_CONTIGUOUS | NPY_ALIGNED);
	ymask = PyArray_FROM_OTF(ymask, NPY_BOOL, NPY_F_CONTIGUOUS | NPY_ALIGNED);

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
		pair<ArrayXXd, ArrayXXd> dataPair;
		
		if(PyArray_NDIM(img) > 2)
			dataPair = generateDataFromImage(
				PyArray_ToArraysXXd(img),
				PyArray_ToMatrixXb(xmask),
				PyArray_ToMatrixXb(ymask),
				num_samples);
		else
			dataPair = generateDataFromImage(
				PyArray_ToMatrixXd(img),
				PyArray_ToMatrixXb(xmask),
				PyArray_ToMatrixXb(ymask),
				num_samples);

		PyObject* xvalues = PyArray_FromMatrixXd(dataPair.first);
		PyObject* yvalues = PyArray_FromMatrixXd(dataPair.second);

		PyObject* data = Py_BuildValue("(OO)",
			xvalues,
			yvalues,
			PyArray_FromMatrixXd(dataPair.second));

		Py_DECREF(img);
		Py_DECREF(xvalues);
		Py_DECREF(yvalues);
		Py_DECREF(xmask);
		Py_DECREF(ymask);

		return data;

	} catch(Exception exception) {
		Py_DECREF(img);
		Py_DECREF(xmask);
		Py_DECREF(ymask);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	return 0;
}



PyObject* sample_image(PyObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"img", "model", "xmask", "ymask", "preconditioner", 0};

	PyObject* img;
	PyObject* model;
	PyObject* xmask;
	PyObject* ymask;
	PyObject* preconditioner = 0;

	if(!PyArg_ParseTupleAndKeywords(args, kwds, "OOOO|O", const_cast<char**>(kwlist),
		&img, &model, &xmask, &ymask, &preconditioner))
		return 0;

	// make sure data is stored in NumPy array
	img = PyArray_FROM_OTF(img, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
	xmask = PyArray_FROM_OTF(xmask, NPY_BOOL, NPY_F_CONTIGUOUS | NPY_ALIGNED);
	ymask = PyArray_FROM_OTF(ymask, NPY_BOOL, NPY_F_CONTIGUOUS | NPY_ALIGNED);

	// TODO: make sure preconditioner is of type TransformObject or similar
	// TODO: make sure that model is of type CDObject or similar
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
		PyObject* imgSample;
		if(PyArray_NDIM(img) > 2) {
			if(preconditioner) {
				imgSample = PyArray_FromArraysXXd(
					sampleImage(
						PyArray_ToArraysXXd(img),
						cd,
						PyArray_ToMatrixXb(xmask),
						PyArray_ToMatrixXb(ymask),
						*reinterpret_cast<TransformObject*>(preconditioner)->transform));
			} else {
				imgSample = PyArray_FromArraysXXd(
					sampleImage(
						PyArray_ToArraysXXd(img),
						cd,
						PyArray_ToMatrixXb(xmask),
						PyArray_ToMatrixXb(ymask)));
			}
		} else {
			if(preconditioner) {
				imgSample = PyArray_FromMatrixXd(
					sampleImage(
						PyArray_ToMatrixXd(img),
						cd,
						PyArray_ToMatrixXb(xmask),
						PyArray_ToMatrixXb(ymask),
						*reinterpret_cast<TransformObject*>(preconditioner)->transform));
			} else {
				imgSample = PyArray_FromMatrixXd(
					sampleImage(
						PyArray_ToMatrixXd(img),
						cd,
						PyArray_ToMatrixXb(xmask),
						PyArray_ToMatrixXb(ymask)));
			}
		}

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



PyObject* shuffle(PyObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"img", 0};

	PyObject* img;

	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O", const_cast<char**>(kwlist), &img))
		return 0;

	// make sure data is stored in NumPy array
	img = PyArray_FROM_OTF(img, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

	if(!img) {
		PyErr_SetString(PyExc_TypeError, "Image has to be stored in an array.");
		return 0;
	}

	if(PyArray_NDIM(img) != 3) {
		PyErr_SetString(PyExc_TypeError, "Image should be three-dimensional.");
		return 0;
	}

	vector<ArrayXXd> channels = PyArray_ToArraysXXd(img);
	vector<ArrayXXd> channelsShuffled;

	for(int i = channels.size() - 1; i >= 0; --i)
		channelsShuffled.push_back(channels[i]);

	return PyArray_FromArraysXXd(channelsShuffled);
}
