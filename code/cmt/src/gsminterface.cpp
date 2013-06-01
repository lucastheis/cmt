#include "gsminterface.h"
#include "distributioninterface.h"

#include "exception.h"
using CMT::Exception;

int GSM_init(GSMObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"dim_in", "num_scales", 0};

	int dim_in = 1;
	int num_scales = 6;

	if(!PyArg_ParseTupleAndKeywords(args, kwds, "|ii", const_cast<char**>(kwlist),
		&dim_in, &num_scales))
		return -1;

	try {
		self->gsm = new GSM(dim_in, num_scales);
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	return 0;
}



PyObject* GSM_mean(GSMObject* self, void*) {
	PyObject* array = PyArray_FromMatrixXd(self->gsm->mean());

	// make array immutable
	reinterpret_cast<PyArrayObject*>(array)->flags &= ~NPY_WRITEABLE;

	return array;
}



int GSM_set_mean(GSMObject* self, PyObject* value, void*) {
	value = PyArray_FROM_OTF(value, NPY_DOUBLE, NPY_IN_ARRAY);

	if(!value) {
		PyErr_SetString(PyExc_TypeError, "Scales should be of type `ndarray`.");
		return -1;
	}

	try {
		self->gsm->setMean(PyArray_ToMatrixXd(value));
	} catch(Exception exception) {
		Py_DECREF(value);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	Py_DECREF(value);

	return 0;
}



PyObject* GSM_priors(GSMObject* self, void*) {
	PyObject* array = PyArray_FromMatrixXd(self->gsm->priors());

	// make array immutable
	reinterpret_cast<PyArrayObject*>(array)->flags &= ~NPY_WRITEABLE;

	return array;
}



int GSM_set_priors(GSMObject* self, PyObject* value, void*) {
	value = PyArray_FROM_OTF(value, NPY_DOUBLE, NPY_IN_ARRAY);

	if(!value) {
		PyErr_SetString(PyExc_TypeError, "Scales should be of type `ndarray`.");
		return -1;
	}

	try {
		self->gsm->setPriors(PyArray_ToMatrixXd(value));
	} catch(Exception exception) {
		Py_DECREF(value);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	Py_DECREF(value);

	return 0;
}



PyObject* GSM_scales(GSMObject* self, void*) {
	PyObject* array = PyArray_FromMatrixXd(self->gsm->scales());

	// make array immutable
	reinterpret_cast<PyArrayObject*>(array)->flags &= ~NPY_WRITEABLE;

	return array;
}



int GSM_set_scales(GSMObject* self, PyObject* value, void*) {
	value = PyArray_FROM_OTF(value, NPY_DOUBLE, NPY_IN_ARRAY);

	if(!value) {
		PyErr_SetString(PyExc_TypeError, "Scales should be of type `ndarray`.");
		return -1;
	}

	try {
		self->gsm->setScales(PyArray_ToMatrixXd(value));
	} catch(Exception exception) {
		Py_DECREF(value);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	Py_DECREF(value);

	return 0;
}



PyObject* GSM_covariance(GSMObject* self, void*) {
	PyObject* array = PyArray_FromMatrixXd(self->gsm->covariance());

	// make array immutable
	reinterpret_cast<PyArrayObject*>(array)->flags &= ~NPY_WRITEABLE;

	return array;
}



int GSM_set_covariance(GSMObject* self, PyObject* value, void*) {
	value = PyArray_FROM_OTF(value, NPY_DOUBLE, NPY_IN_ARRAY);

	if(!value) {
		PyErr_SetString(PyExc_TypeError, "Scales should be of type `ndarray`.");
		return -1;
	}

	try {
		self->gsm->setCovariance(PyArray_ToMatrixXd(value));
	} catch(Exception exception) {
		Py_DECREF(value);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	Py_DECREF(value);

	return 0;
}



GSM::Parameters* PyObject_ToGSMParameters(PyObject* parameters) {
	GSM::Parameters* params = new GSM::Parameters;

	if(parameters && parameters != Py_None) {
//		PyObject* callback = PyDict_GetItemString(parameters, "callback");
//		if(callback)
//			if(PyCallable_Check(callback))
//				params->callback = new CallbackInterface(&GSM_type, callback);
//			else if(callback != Py_None)
//				throw Exception("callback should be a function or callable object.");

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



PyObject* GSM_train(GSMObject* self, PyObject* args, PyObject* kwds) {
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

	try {
		bool converged;

		GSM::Parameters* params = PyObject_ToGSMParameters(parameters);

		converged = self->gsm->train(PyArray_ToMatrixXd(data), *params);

		delete params;

		Py_DECREF(data);

		if(converged) {
			Py_INCREF(Py_True);
			return Py_True;
		} else {
			Py_INCREF(Py_False);
			return Py_False;
		}
	} catch(Exception exception) {
		Py_DECREF(data);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	return 0;
}



PyObject* GSM_reduce(GSMObject* self, PyObject*) {
	// constructor arguments
	PyObject* args = Py_BuildValue("(ii)",
		self->gsm->dim(),
		self->gsm->numScales());

	PyObject* mean = GSM_mean(self, 0);
	PyObject* priors = GSM_priors(self, 0);
	PyObject* scales = GSM_scales(self, 0);
	PyObject* covariance = GSM_covariance(self, 0);

	PyObject* state = Py_BuildValue("(OOOO)", mean, priors, scales, covariance);
	PyObject* result = Py_BuildValue("(OOO)", Py_TYPE(self), args, state);

	Py_DECREF(mean);
	Py_DECREF(priors);
	Py_DECREF(scales);
	Py_DECREF(covariance);
	Py_DECREF(args);
	Py_DECREF(state);

	return result;
}



PyObject* GSM_setstate(GSMObject* self, PyObject* state) {
	PyObject* mean;
	PyObject* priors;
	PyObject* scales;
	PyObject* covariance;

	if(!PyArg_ParseTuple(state, "(OOOO)", &mean, &priors, &scales, &covariance))
		return 0;

	try {
		GSM_set_mean(self, mean, 0);
		GSM_set_priors(self, priors, 0);
		GSM_set_scales(self, scales, 0);
		GSM_set_covariance(self, covariance, 0);
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	Py_INCREF(Py_None);
	return Py_None;
}
