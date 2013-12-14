#include "gsminterface.h"
#include "distributioninterface.h"

#include "cmt/utils"
using CMT::Exception;

const char* GSM_doc =
	"An implementation of a finite Gaussian scale mixture.\n"
	"\n"
	"$$p(\\mathbf{x}) = \\sum_k \\pi_k \\mathcal{N}(\\mathbf{x}; \\mu_k, \\lambda_k^{-1} \\boldsymbol{\\Sigma})$$\n"
	"\n"
	"This is a mixture of Gaussians which all share the same basic covariance\n"
	"structure $\\boldsymbol{\\Sigma}$ scaled by the precision variables $\\lambda_k$.\n"
	"The prior probability of selecting component $k$ for generating samples is $\\pi_k$.\n"
	"\n"
	"@type  dim: C{int}\n"
	"@param dim: dimensionality of the distribution\n"
	"\n"
	"@type  num_scales: C{int}\n"
	"@param num_scales: number of precision scale variables";

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



PyObject* GSM_num_scales(GSMObject* self, void*) {
	return PyInt_FromLong(self->gsm->numScales());
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
