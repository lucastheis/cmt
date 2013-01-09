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



PyObject* Transform_dim_in(TransformObject* self, PyObject*, void*) {
	return PyInt_FromLong(self->transform->dimIn());
}



PyObject* Transform_dim_out(TransformObject* self, PyObject*, void*) {
	return PyInt_FromLong(self->transform->dimOut());
}



int AffineTransform_init(AffineTransformObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"matrix", "vector", 0};

	PyObject* matrix;
	PyObject* vector;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "OO", const_cast<char**>(kwlist), &matrix, &vector))
		return -1;

	matrix = PyArray_FROM_OTF(matrix, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
	vector = PyArray_FROM_OTF(vector, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

	if(!matrix) {
		PyErr_SetString(PyExc_TypeError, "Matrix should be of type `ndarray`.");
		return -1;
	}

	if(!vector) {
		PyErr_SetString(PyExc_TypeError, "Vector should be of type `ndarray`.");
		return -1;
	}

	MatrixXd vec = PyArray_ToMatrixXd(vector);

	if(vec.cols() != 1) {
		PyErr_SetString(PyExc_TypeError, "Offset needs to be a vector.");
		return -1;
	}

	// create actual instance
	self->transform = new MCM::AffineTransform(PyArray_ToMatrixXd(matrix), vec);

	Py_DECREF(matrix);

	return 0;
}



PyObject* AffineTransform_A(AffineTransformObject* self, PyObject*, void*) {
	PyObject* array = PyArray_FromMatrixXd(self->transform->matrix());

	// make array immutable
	reinterpret_cast<PyArrayObject*>(array)->flags &= ~NPY_WRITEABLE;

	return array;
}



int AffineTransform_set_A(AffineTransformObject* self, PyObject* value, void*) {
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



PyObject* AffineTransform_b(AffineTransformObject* self, PyObject*, void*) {
	PyObject* array = PyArray_FromMatrixXd(self->transform->vector());

	// make array immutable
	reinterpret_cast<PyArrayObject*>(array)->flags &= ~NPY_WRITEABLE;

	return array;
}



int AffineTransform_set_b(AffineTransformObject* self, PyObject* value, void*) {
	if(!PyArray_Check(value)) {
		PyErr_SetString(PyExc_TypeError, "Matrix should be of type `ndarray`.");
		return -1;
	}

	try {
		MatrixXd vec = PyArray_ToMatrixXd(value);

		if(vec.cols() != 1)
			throw Exception("Offset needs to be a vector."); 

		self->transform->setVector(vec);

	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	return 0;
}



PyObject* AffineTransform_reduce(AffineTransformObject* self, PyObject*, PyObject*) {
	PyObject* matrix = AffineTransform_A(self, 0, 0);
	PyObject* vector = AffineTransform_b(self, 0, 0);
	PyObject* args = Py_BuildValue("(OO)", matrix, vector);
	PyObject* state = Py_BuildValue("()");
	PyObject* result = Py_BuildValue("OOO", self->ob_type, args, state);

	Py_DECREF(matrix);
	Py_DECREF(vector);
	Py_DECREF(args);
	Py_DECREF(state);

	return result;
}



PyObject* AffineTransform_setstate(AffineTransformObject* self, PyObject* state, PyObject*) {
	Py_INCREF(Py_None);
	return Py_None;
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



PyObject* LinearTransform_reduce(AffineTransformObject* self, PyObject*, PyObject*) {
	PyObject* matrix = AffineTransform_A(self, 0, 0);
	PyObject* args = Py_BuildValue("(O)", matrix);
	PyObject* state = Py_BuildValue("()");
	PyObject* result = Py_BuildValue("OOO", self->ob_type, args, state);

	Py_DECREF(matrix);
	Py_DECREF(args);
	Py_DECREF(state);

	return result;
}



PyObject* LinearTransform_setstate(AffineTransformObject* self, PyObject* state, PyObject*) {
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



PyObject* WhiteningTransform_reduce(AffineTransformObject* self, PyObject*, PyObject*) {
	PyObject* dummy = PyArray_FromMatrixXd(MatrixXd::Identity(2, 2));
	PyObject* matrix = AffineTransform_A(self, 0, 0);
	PyObject* args = Py_BuildValue("(O)", dummy);
	PyObject* state = Py_BuildValue("(O)", matrix);
	PyObject* result = Py_BuildValue("OOO", self->ob_type, args, state);

	Py_DECREF(dummy);
	Py_DECREF(matrix);
	Py_DECREF(args);
	Py_DECREF(state);

	return result;
}



PyObject* WhiteningTransform_setstate(AffineTransformObject* self, PyObject* state, PyObject*) {
	PyObject* matrix;

	if(!PyArg_ParseTuple(state, "(O)", &matrix))
		return 0;

	try {
		AffineTransform_set_A(self, matrix, 0);
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	Py_INCREF(Py_None);
	return Py_None;
}



int PCATransform_init(PCATransformObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"data", "num_pcs", 0};

	PyObject* data;
	int num_pcs = -1;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O|i", const_cast<char**>(kwlist), &data, &num_pcs))
		return -1;

	data = PyArray_FROM_OTF(data, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

	if(!data) {
		PyErr_SetString(PyExc_TypeError, "Data should be of type `ndarray`.");
		return -1;
	}

	// create actual instance
	self->transform = new MCM::PCATransform(PyArray_ToMatrixXd(data), num_pcs);

	Py_DECREF(data);

	return 0;
}



PyObject* PCATransform_eigenvalues(PCATransformObject* self, PyObject*, PyObject*) {
	PyObject* array = PyArray_FromMatrixXd(self->transform->eigenvalues());

	// make array immutable
	reinterpret_cast<PyArrayObject*>(array)->flags &= ~NPY_WRITEABLE;

	return array;
}
