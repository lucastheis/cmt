#include "distributioninterface.h"
#include "mixtureinterface.h"
#include "pyutils.h"

#include "exception.h"
using CMT::Exception;

int Mixture_init(MixtureObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"dim", 0};

	int dim;

	if(!PyArg_ParseTupleAndKeywords(args, kwds, "i", const_cast<char**>(kwlist), &dim))
		return -1;

	try {
		self->mixture = new Mixture(dim);
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	return 0;
}



Mixture::Parameters* PyObject_ToMixtureParameters(PyObject* parameters) {
	Mixture::Parameters* params = new Mixture::Parameters;

	if(parameters && parameters != Py_None) {
		PyObject* max_iter = PyDict_GetItemString(parameters, "max_iter");
		if(max_iter)
			if(PyInt_Check(max_iter))
				params->maxIter = PyInt_AsLong(max_iter);
			else if(PyFloat_Check(max_iter))
				params->maxIter = static_cast<int>(PyFloat_AsDouble(max_iter));
			else
				throw Exception("max_iter should be of type `int`.");
	}
	
	return params;
}



PyObject* Mixture_train(MixtureObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"data", "parameters", 0};

	PyObject* data;
	PyObject* parameters = 0;

	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O|O", const_cast<char**>(kwlist),
		&data, &parameters))
		return 0;

	data = PyArray_FROM_OTF(data, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
	
	if(!data) {
		PyErr_SetString(PyExc_TypeError, "Data should be stored in a Numpy array.");
		return 0;
	}

	bool converged;

	try {
		Mixture::Parameters* params = PyObject_ToMixtureParameters(parameters);
		converged = self->mixture->train(PyArray_ToMatrixXd(data), *params);
		delete params;
	} catch(Exception exception) {
		Py_DECREF(data);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	Py_DECREF(data);

	return 0;
}



PyObject* Mixture_subscript(MixtureObject* self, PyObject* key) {
	if(!PyInt_Check(key)) {
		PyErr_SetString(PyExc_TypeError, "Index must be an integer.");
		return 0;
	}

	PyObject* component = Distribution_new(&MixtureComponent_type, 0, 0);
	reinterpret_cast<MixtureComponentObject*>(component)->component =
		self->mixture->operator[](PyInt_AsLong(key));
	reinterpret_cast<MixtureComponentObject*>(component)->owner = false;

	Py_INCREF(component);
	return component;
}



PyObject* Mixture_add_component(MixtureObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"component", 0};

	PyObject* component;

	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O!", const_cast<char**>(kwlist),
		&MixtureComponent_type, &component))
		return 0;

	return 0;
}



//Mixture::Component::Parameters* PyObject_ToMixtureComponentParameters(PyObject* parameters) {
//	Mixture::Component::Parameters* params = new Mixture::Component::Parameters;
//
//	if(parameters && parameters != Py_None) {
//		PyObject* max_iter = PyDict_GetItemString(parameters, "max_iter");
//		if(max_iter)
//			if(PyInt_Check(max_iter))
//				params->maxIter = PyInt_AsLong(max_iter);
//			else if(PyFloat_Check(max_iter))
//				params->maxIter = static_cast<int>(PyFloat_AsDouble(max_iter));
//			else
//				throw Exception("max_iter should be of type `int`.");
//	}
//
//	return params;
//}



int MixtureComponent_init(MixtureComponentObject*, PyObject*, PyObject*) {
	PyErr_SetString(PyExc_NotImplementedError, "This is an abstract class.");
	return -1;
}



PyObject* MixtureComponent_train(MixtureComponentObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"data", "parameters", 0};

	PyObject* data;
	PyObject* parameters = 0;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O|O", const_cast<char**>(kwlist),
		&data, &parameters))
		return 0;

	// make sure data is stored in NumPy array
	data = PyArray_FROM_OTF(data, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

	if(!data) {
		PyErr_SetString(PyExc_TypeError, "Data has to be stored in NumPy array.");
		return 0;
	}

	bool converged;

	try {
//		Mixture::Component::Parameters* params = PyObject_ToMixtureComponentParameters(parameters);
//		converged = self->gsm->train(PyArray_ToMatrixXd(data), *params);
//		delete params;
		converged = self->component->train(PyArray_ToMatrixXd(data));
	} catch(Exception exception) {
		Py_DECREF(data);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	Py_DECREF(data);

	if(converged) {
		Py_INCREF(Py_True);
		return Py_True;
	}

	Py_INCREF(Py_False);
	return Py_False;
}
