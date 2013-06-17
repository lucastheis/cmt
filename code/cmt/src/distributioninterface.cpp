#include "distributioninterface.h"
#include "Eigen/Core"

#include "exception.h"
using CMT::Exception;

PyObject* Distribution_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
	PyObject* self = type->tp_alloc(type, 0);

	if(self) {
		reinterpret_cast<DistributionObject*>(self)->dist = 0;
		reinterpret_cast<DistributionObject*>(self)->owner = true;
	}

	return self;
}



const char* Distribution_doc =
	"Abstract base class for distributions.\n";

int Distribution_init(DistributionObject*, PyObject*, PyObject*) {
	PyErr_SetString(PyExc_NotImplementedError, "This is an abstract class.");
	return -1;
}



void Distribution_dealloc(DistributionObject* self) {
	// delete actual instance
	if(self->owner)
		delete self->dist;

	// delete DistributionObject
	self->ob_type->tp_free(reinterpret_cast<PyObject*>(self));
}



PyObject* Distribution_dim(DistributionObject* self, PyObject*, void*) {
	return PyInt_FromLong(self->dist->dim());
}



const char* Distribution_sample_doc =
	"sample(self, num_samples)\n"
	"\n"
	"Generates samples from the distribution.\n"
	"\n"
	"@type  num_samples: int\n"
	"@param num_samples: the number of samples drawn from the distribution\n"
	"\n"
	"@rtype: ndarray\n"
	"@return: samples";

PyObject* Distribution_sample(DistributionObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"num_samples", 0};

	int num_samples;

	if(!PyArg_ParseTupleAndKeywords(args, kwds, "i", const_cast<char**>(kwlist), &num_samples))
		return 0;

	try {
		return PyArray_FromMatrixXd(self->dist->sample(num_samples));
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	return 0;
}



const char* Distribution_loglikelihood_doc =
	"loglikelihood(self, data)\n"
	"\n"
	"Computes the log-likelihood for the given data points in nats.\n"
	"\n"
	"@type  data: ndarray\n"
	"@param data: data points stored in columns\n"
	"\n"
	"@rtype: ndarray\n"
	"@return: log-likelihood of the model evaluated for each data point";

PyObject* Distribution_loglikelihood(DistributionObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"data", 0};

	PyObject* data;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O", const_cast<char**>(kwlist), &data))
		return 0;

	// make sure data is stored in NumPy array
	data = PyArray_FROM_OTF(data, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

	if(!data) {
		PyErr_SetString(PyExc_TypeError, "Data has to be stored in NumPy arrays.");
		return 0;
	}

	try {
		PyObject* result = PyArray_FromMatrixXd(
			self->dist->logLikelihood(
				PyArray_ToMatrixXd(data)));
		Py_DECREF(data);
		return result;
	} catch(Exception exception) {
		Py_DECREF(data);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	return 0;
}



const char* Distribution_evaluate_doc =
	"evaluate(self, data)\n"
	"\n"
	"Computes the average negative log-likelihood for the given data points "
	"in bits per component (smaller is better).\n"
	"\n"
	"@type  data: ndarray\n"
	"@param data: data stored in columns\n"
	"\n"
	"@rtype: double\n"
	"@return: performance in bits per component";

PyObject* Distribution_evaluate(DistributionObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"data", 0};

	PyObject* data;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O", const_cast<char**>(kwlist), &data))
		return 0;

	// make sure data is stored in NumPy array
	data = PyArray_FROM_OTF(data, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

	if(!data) {
		PyErr_SetString(PyExc_TypeError, "Data has to be stored in NumPy arrays.");
		return 0;
	}

	try {
		double result = 
			self->dist->evaluate(
				PyArray_ToMatrixXd(data));
		Py_DECREF(data);
		return PyFloat_FromDouble(result);
	} catch(Exception exception) {
		Py_DECREF(data);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	return 0;
}
