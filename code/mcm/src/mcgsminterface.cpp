#include "mcgsminterface.h"
#include "Eigen/Core"
#include "exception.h"

using namespace Eigen;

PyObject* MCGSM_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
	PyObject* self = type->tp_alloc(type, 0);

	if(self)
		reinterpret_cast<MCGSMObject*>(self)->mcgsm = 0;

	return self;
}



int MCGSM_init(MCGSMObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"dim_in", "dim_out", "num_components", "num_scales", "num_features", 0};
	int dim_in;
	int dim_out = 1;
	int num_components = 8;
	int num_scales = 6;
	int num_features = 0;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "i|iiii", const_cast<char**>(kwlist),
		&dim_in, &dim_out, &num_components, &num_scales, &num_features))
		return -1;

	if(!num_features)
		num_features = dim_in;

	// create actual GSM instance
	self->mcgsm = new MCGSM(dim_in, dim_out, num_components, num_scales, num_features);

	return 0;
}



void MCGSM_dealloc(MCGSMObject* self) {
	// delete actual GSM instance
	delete self->mcgsm;

	// delete GSM object
	self->ob_type->tp_free(reinterpret_cast<PyObject*>(self));
}



PyObject* MCGSM_dim_in(MCGSMObject* self, PyObject*, void*) {
	return PyInt_FromLong(self->mcgsm->dimIn());
}



PyObject* MCGSM_dim_out(MCGSMObject* self, PyObject*, void*) {
	return PyInt_FromLong(self->mcgsm->dimOut());
}



PyObject* MCGSM_num_components(MCGSMObject* self, PyObject*, void*) {
	return PyInt_FromLong(self->mcgsm->numComponents());
}



PyObject* MCGSM_num_scales(MCGSMObject* self, PyObject*, void*) {
	return PyInt_FromLong(self->mcgsm->numScales());
}



PyObject* MCGSM_num_features(MCGSMObject* self, PyObject*, void*) {
	return PyInt_FromLong(self->mcgsm->numFeatures());
}



PyObject* MCGSM_priors(MCGSMObject* self, PyObject*, void*) {
	PyObject* array = PyArray_FromMatrixXd(self->mcgsm->priors());

	// make array immutable
	reinterpret_cast<PyArrayObject*>(array)->flags &= ~NPY_WRITEABLE;

	return array;
}



int MCGSM_set_priors(MCGSMObject* self, PyObject* value, void*) {
	value = PyArray_FROM_OTF(value, NPY_DOUBLE, NPY_IN_ARRAY);

	if(!value) {
		PyErr_SetString(PyExc_TypeError, "Priors should be of type `ndarray`.");
		return -1;
	}

	try {
		self->mcgsm->setPriors(PyArray_ToMatrixXd(value));
	} catch(Exception exception) {
		Py_DECREF(value);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	Py_DECREF(value);

	return 0;
}



PyObject* MCGSM_scales(MCGSMObject* self, PyObject*, void*) {
	PyObject* array = PyArray_FromMatrixXd(self->mcgsm->scales());

	// make array immutable
	reinterpret_cast<PyArrayObject*>(array)->flags &= ~NPY_WRITEABLE;

	return array;
}



int MCGSM_set_scales(MCGSMObject* self, PyObject* value, void*) {
	value = PyArray_FROM_OTF(value, NPY_DOUBLE, NPY_IN_ARRAY);

	if(!value) {
		PyErr_SetString(PyExc_TypeError, "Scales should be of type `ndarray`.");
		return -1;
	}

	try {
		self->mcgsm->setScales(PyArray_ToMatrixXd(value));
	} catch(Exception exception) {
		Py_DECREF(value);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	Py_DECREF(value);

	return 0;
}



PyObject* MCGSM_weights(MCGSMObject* self, PyObject*, void*) {
	PyObject* array = PyArray_FromMatrixXd(self->mcgsm->weights());

	// make array immutable
	reinterpret_cast<PyArrayObject*>(array)->flags &= ~NPY_WRITEABLE;

	return array;
}



int MCGSM_set_weights(MCGSMObject* self, PyObject* value, void*) {
	value = PyArray_FROM_OTF(value, NPY_DOUBLE, NPY_IN_ARRAY);

	if(!value) {
		PyErr_SetString(PyExc_TypeError, "Weights should be of type `ndarray`.");
		return -1;
	}

	try {
		self->mcgsm->setWeights(PyArray_ToMatrixXd(value));
	} catch(Exception exception) {
		Py_DECREF(value);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	Py_DECREF(value);

	return 0;
}



PyObject* MCGSM_features(MCGSMObject* self, PyObject*, void*) {
	PyObject* array = PyArray_FromMatrixXd(self->mcgsm->features());

	// make array immutable
	reinterpret_cast<PyArrayObject*>(array)->flags &= ~NPY_WRITEABLE;

	return array;
}



int MCGSM_set_features(MCGSMObject* self, PyObject* value, void*) {
	value = PyArray_FROM_OTF(value, NPY_DOUBLE, NPY_IN_ARRAY);

	if(!value) {
		PyErr_SetString(PyExc_TypeError, "Features should be of type `ndarray`.");
		return -1;
	}

	try {
		self->mcgsm->setFeatures(PyArray_ToMatrixXd(value));
	} catch(Exception exception) {
		Py_DECREF(value);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	Py_DECREF(value);

	return 0;
}



PyObject* MCGSM_cholesky_factors(MCGSMObject* self, PyObject*, void*) {
	vector<MatrixXd> choleskyFactors = self->mcgsm->choleskyFactors();

	PyObject* list = PyList_New(choleskyFactors.size());

 	for(unsigned int i = 0; i < choleskyFactors.size(); ++i) {
		// create immutable array
		PyObject* array = PyArray_FromMatrixXd(choleskyFactors[i]);
		reinterpret_cast<PyArrayObject*>(array)->flags &= ~NPY_WRITEABLE;
 
 		// add array to list
 		PyList_SetItem(list, i, array);
 	}

	return list;
}



int MCGSM_set_cholesky_factors(MCGSMObject* self, PyObject* value, void*) {
	const char* kwlist[] = {"cholesky_factors", 0};

	if(!PyList_Check(value)) {
		PyErr_SetString(PyExc_TypeError, "Cholesky factors should be given in a list.");
		return 0;
	}

	try {
		vector<MatrixXd> choleskyFactors;

		for(Py_ssize_t i = 0; i < PyList_Size(value); ++i) {
 			PyObject* array = PyList_GetItem(value, i);

 			array = PyArray_FROM_OTF(array, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
 
 			if(!array) {
 				PyErr_SetString(PyExc_TypeError, "Cholesky factors should be of type `ndarray`.");
 				return 0;
 			}

			choleskyFactors.push_back(PyArray_ToMatrixXd(array));

			// remove reference created by PyArray_FROM_OTF
			Py_DECREF(array);
		}

		self->mcgsm->setCholeskyFactors(choleskyFactors);

	} catch(Exception exception) {
		PyErr_SetString(PyExc_TypeError, exception.message());
		return 0;
	}

	return 0;
}



PyObject* MCGSM_predictors(MCGSMObject* self, PyObject*, void*) {
	vector<MatrixXd> predictors = self->mcgsm->predictors();

	PyObject* list = PyList_New(predictors.size());

 	for(unsigned int i = 0; i < predictors.size(); ++i) {
		// create immutable array
		PyObject* array = PyArray_FromMatrixXd(predictors[i]);
		reinterpret_cast<PyArrayObject*>(array)->flags &= ~NPY_WRITEABLE;
 
 		// add array to list
 		PyList_SetItem(list, i, array);
 	}

	return list;
}



int MCGSM_set_predictors(MCGSMObject* self, PyObject* value, void*) {
	const char* kwlist[] = {"predictors", 0};

	if(!PyList_Check(value)) {
		PyErr_SetString(PyExc_TypeError, "Predictors should be given in a list.");
		return 0;
	}

	try {
		vector<MatrixXd> predictors;

		for(Py_ssize_t i = 0; i < PyList_Size(value); ++i) {
 			PyObject* array = PyList_GetItem(value, i);

 			array = PyArray_FROM_OTF(array, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
 
 			if(!array) {
 				PyErr_SetString(PyExc_TypeError, "Predictors should be of type `ndarray`.");
 				return 0;
 			}

			predictors.push_back(PyArray_ToMatrixXd(array));

			// remove reference created by PyArray_FROM_OTF
			Py_DECREF(array);
		}

		self->mcgsm->setPredictors(predictors);

	} catch(Exception exception) {
		PyErr_SetString(PyExc_TypeError, exception.message());
		return 0;
	}

	return 0;
}



PyObject* MCGSM_normalize(MCGSMObject* self, PyObject*, PyObject*) {
	try {
		self->mcgsm->normalize();
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	Py_INCREF(Py_None);
	return Py_None;
}



PyObject* MCGSM_train(MCGSMObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"input", "output", "max_iter", "tol", 0};

	PyObject* input;
	PyObject* output;
	int max_iter = 100;
	double tol = 1e-5;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "OO|id", const_cast<char**>(kwlist), &input, &output, &max_iter, &tol))
		return 0;

	// make sure data is stored in NumPy array
	input = PyArray_FROM_OTF(input, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
	output = PyArray_FROM_OTF(output, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

	if(!input || !output) {
		PyErr_SetString(PyExc_TypeError, "Data has to be stored in NumPy arrays.");
		return 0;
	}

	try {
		if(self->mcgsm->train(PyArray_ToMatrixXd(input), PyArray_ToMatrixXd(output), max_iter, tol)) {
			Py_DECREF(input);
			Py_DECREF(output);
			Py_INCREF(Py_True);
			return Py_True;
		} else {
			Py_DECREF(input);
			Py_DECREF(output);
			Py_INCREF(Py_False);
			return Py_False;
		}
	} catch(Exception exception) {
		Py_DECREF(input);
		Py_DECREF(output);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	return 0;
}



PyObject* MCGSM_posterior(MCGSMObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"input", "output", 0};

	PyObject* input;
	PyObject* output;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "OO", const_cast<char**>(kwlist), &input, &output))
		return 0;

	// make sure data is stored in NumPy array
	input = PyArray_FROM_OTF(input, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
	output = PyArray_FROM_OTF(output, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

	if(!input || !output) {
		PyErr_SetString(PyExc_TypeError, "Data has to be stored in NumPy arrays.");
		return 0;
	}

	try {
		PyObject* result = PyArray_FromMatrixXd(
			self->mcgsm->posterior(PyArray_ToMatrixXd(input), PyArray_ToMatrixXd(output)));
		Py_DECREF(input);
		Py_DECREF(output);
		return result;
	} catch(Exception exception) {
		Py_DECREF(input);
		Py_DECREF(output);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}
}



PyObject* MCGSM_sample(MCGSMObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"input", 0};

	PyObject* input;

	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O", const_cast<char**>(kwlist), &input))
		return 0;

	input = PyArray_FROM_OTF(input, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

	if(!input) {
		PyErr_SetString(PyExc_TypeError, "Data has to be stored in a NumPy array.");
		return 0;
	}

	try {
		PyObject* result = PyArray_FromMatrixXd(self->mcgsm->sample(PyArray_ToMatrixXd(input)));
		Py_DECREF(input);
		return result;
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}
}



PyObject* MCGSM_sample_posterior(MCGSMObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"input", "output", 0};

	PyObject* input;
	PyObject* output;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "OO", const_cast<char**>(kwlist), &input, &output))
		return 0;

	// make sure data is stored in NumPy array
	input = PyArray_FROM_OTF(input, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
	output = PyArray_FROM_OTF(output, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

	if(!input || !output) {
		PyErr_SetString(PyExc_TypeError, "Data has to be stored in NumPy arrays.");
		return 0;
	}

	try {
		PyObject* result = PyArray_FromMatrixXd(
			self->mcgsm->samplePosterior(PyArray_ToMatrixXd(input), PyArray_ToMatrixXd(output)));
		Py_DECREF(input);
		Py_DECREF(output);
		return result;
	} catch(Exception exception) {
		Py_DECREF(input);
		Py_DECREF(output);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}
}



PyObject* MCGSM_loglikelihood(MCGSMObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"input", "output", 0};

	PyObject* input;
	PyObject* output;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "OO", const_cast<char**>(kwlist), &input, &output))
		return 0;

	// make sure data is stored in NumPy array
	input = PyArray_FROM_OTF(input, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
	output = PyArray_FROM_OTF(output, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

	if(!input || !output) {
		PyErr_SetString(PyExc_TypeError, "Data has to be stored in NumPy arrays.");
		return 0;
	}

	try {
		PyObject* result = PyArray_FromMatrixXd(
			self->mcgsm->logLikelihood(PyArray_ToMatrixXd(input), PyArray_ToMatrixXd(output)));
		Py_DECREF(input);
		Py_DECREF(output);
		return result;
	} catch(Exception exception) {
		Py_DECREF(input);
		Py_DECREF(output);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}
}



PyObject* MCGSM_reduce(MCGSMObject* self, PyObject*, PyObject*) {
	PyObject* args = Py_BuildValue("(iiiii)", 
		self->mcgsm->dimIn(),
		self->mcgsm->dimOut(),
		self->mcgsm->numComponents(),
		self->mcgsm->numScales(),
		self->mcgsm->numFeatures());

	PyObject* priors = MCGSM_priors(self, 0, 0);
	PyObject* scales = MCGSM_scales(self, 0, 0);
	PyObject* weights = MCGSM_weights(self, 0, 0);
	PyObject* features = MCGSM_features(self, 0, 0);
	PyObject* cholesky_factors = MCGSM_cholesky_factors(self, 0, 0);
	PyObject* predictors = MCGSM_predictors(self, 0, 0);
	PyObject* state = Py_BuildValue("(OOOOOO)", 
		priors, scales, weights, features, cholesky_factors, predictors);
	Py_DECREF(priors);
	Py_DECREF(scales);
	Py_DECREF(weights);
	Py_DECREF(features);
	Py_DECREF(cholesky_factors);
	Py_DECREF(predictors);

	PyObject* result = Py_BuildValue("OOO", self->ob_type, args, state);
	Py_DECREF(args);
	Py_DECREF(state);

	return result;
}



PyObject* MCGSM_setstate(MCGSMObject* self, PyObject* state, PyObject*) {
	PyObject* priors;
	PyObject* scales;
	PyObject* weights;
	PyObject* features;
	PyObject* cholesky_factors;
	PyObject* predictors;

	if(!PyArg_ParseTuple(state, "(OOOOOO)",
		&priors, &scales, &weights, &features, &cholesky_factors, &predictors))
		return 0;

	try {
		MCGSM_set_priors(self, priors, 0);
		MCGSM_set_scales(self, scales, 0);
		MCGSM_set_weights(self, weights, 0);
		MCGSM_set_features(self, features, 0);
		MCGSM_set_cholesky_factors(self, cholesky_factors, 0);
		MCGSM_set_predictors(self, predictors, 0);
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	Py_INCREF(Py_None);
	return Py_None;
}