#include "transforminterface.h"
#include "exception.h"

#include "Eigen/Core"
using Eigen::MatrixXd;

PyObject* Transform_call(LinearTransformObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"input", 0};

	PyObject* input;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O", const_cast<char**>(kwlist), &input))
		return 0;

	input = PyArray_FROM_OTF(input, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

	// make sure data is stored in NumPy array
	if(!input) {
		PyErr_SetString(PyExc_TypeError, "Input has to be stored in a NumPy array.");
		return 0;
	}

	try {
		return PyArray_FromMatrixXd(self->transform->operator()(PyArray_ToMatrixXd(input)));
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		Py_DECREF(input);
		return 0;
	}

	return 0;
}



PyObject* Transform_inverse(LinearTransformObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"output", 0};

	PyObject* output;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O", const_cast<char**>(kwlist), &output))
		return 0;

	output = PyArray_FROM_OTF(output, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

	// make sure data is stored in NumPy array
	if(!output) {
		PyErr_SetString(PyExc_TypeError, "Input has to be stored in a NumPy array.");
		return 0;
	}

	try {
		return PyArray_FromMatrixXd(self->transform->inverse(PyArray_ToMatrixXd(output)));
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		Py_DECREF(output);
		return 0;
	}

	return 0;
}



PyObject* Transform_new(PyTypeObject* type, PyObject*, PyObject*) {
	PyObject* self = type->tp_alloc(type, 0);

	if(self)
		reinterpret_cast<TransformObject*>(self)->transform = 0;

	return self;
}



void Transform_dealloc(TransformObject* self) {
	// delete actual instance
	delete self->transform;

	// delete Python object
	self->ob_type->tp_free(reinterpret_cast<PyObject*>(self));
}



int LinearTransform_init(LinearTransformObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"matrix", 0};

	PyObject* matrix;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O", const_cast<char**>(kwlist), &matrix))
		return -1;

	matrix = PyArray_FROM_OTF(matrix, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

	if(!matrix) {
		PyErr_SetString(PyExc_TypeError, "Matrix should be of type `ndarray`.");
		return -1;
	}

	// create actual instance
	self->transform = new MCM::LinearTransform(PyArray_ToMatrixXd(matrix));

	Py_DECREF(matrix);

	return 0;
}



PyObject* LinearTransform_A(LinearTransformObject* self, PyObject*, void*) {
	PyObject* array = PyArray_FromMatrixXd(self->transform->matrix());

	// make array immutable
	reinterpret_cast<PyArrayObject*>(array)->flags &= ~NPY_WRITEABLE;

	return array;
}



int LinearTransform_set_A(LinearTransformObject* self, PyObject* value, void*) {
	if(!PyArray_Check(value)) {
		PyErr_SetString(PyExc_TypeError, "Matrix should be of type `ndarray`.");
		return -1;
	}

	try {
		self->transform->setMatrix(PyArray_ToMatrixXd(value));

	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	return 0;
}



PyObject* LinearTransform_reduce(LinearTransformObject* self, PyObject*, PyObject*) {
	PyObject* matrix = LinearTransform_A(self, 0, 0);
	PyObject* args = Py_BuildValue("(O)", matrix);
	PyObject* state = Py_BuildValue("()");
	PyObject* result = Py_BuildValue("OOO", self->ob_type, args, state);

	Py_DECREF(matrix);
	Py_DECREF(args);
	Py_DECREF(state);

	return result;
}



PyObject* LinearTransform_setstate(LinearTransformObject* self, PyObject* state, PyObject*) {
	Py_INCREF(Py_None);
	return Py_None;
}



int WhiteningTransform_init(WhiteningTransformObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"data", 0};

	PyObject* data;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O", const_cast<char**>(kwlist), &data))
		return -1;

	data = PyArray_FROM_OTF(data, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

	if(!data) {
		PyErr_SetString(PyExc_TypeError, "Data should be of type `ndarray`.");
		return -1;
	}

	// create actual instance
	self->transform = new MCM::WhiteningTransform(PyArray_ToMatrixXd(data));

	Py_DECREF(data);

	return 0;
}



PyObject* WhiteningTransform_reduce(LinearTransformObject* self, PyObject*, PyObject*) {
	PyObject* dummy = PyArray_FromMatrixXd(MatrixXd::Identity(2, 2));
	PyObject* matrix = LinearTransform_A(self, 0, 0);
	PyObject* args = Py_BuildValue("(O)", dummy);
	PyObject* state = Py_BuildValue("(O)", matrix);
	PyObject* result = Py_BuildValue("OOO", self->ob_type, args, state);

	Py_DECREF(dummy);
	Py_DECREF(matrix);
	Py_DECREF(args);
	Py_DECREF(state);

	return result;
}



PyObject* WhiteningTransform_setstate(LinearTransformObject* self, PyObject* state, PyObject*) {
	PyObject* matrix;

	if(!PyArg_ParseTuple(state, "(O)", &matrix))
		return 0;

	try {
		LinearTransform_set_A(self, matrix, 0);
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	Py_INCREF(Py_None);
	return Py_None;
}
