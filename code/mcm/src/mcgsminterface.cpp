#include "mcgsminterface.h"
#include "Eigen/Core"
#include "exception.h"
#include "callbacktrain.h"

using namespace Eigen;

MCGSM::Parameters PyObject_ToParameters(MCGSMObject* self, PyObject* parameters) {
	MCGSM::Parameters params;

	// read parameters from dictionary
	if(parameters) {
		if(!PyDict_Check(parameters))
			throw Exception("Parameters should be stored in a dictionary.");

		PyObject* verbosity = PyDict_GetItemString(parameters, "verbosity");
		if(verbosity)
			if(PyInt_Check(verbosity))
				params.verbosity = PyInt_AsLong(verbosity);
			else if(PyFloat_Check(verbosity))
				params.verbosity = static_cast<int>(PyFloat_AsDouble(verbosity));
			else
				throw Exception("verbosity should be of type `int`.");

		PyObject* max_iter = PyDict_GetItemString(parameters, "max_iter");
		if(max_iter)
			if(PyInt_Check(max_iter))
				params.maxIter = PyInt_AsLong(max_iter);
			else if(PyFloat_Check(max_iter))
				params.maxIter = static_cast<int>(PyFloat_AsDouble(max_iter));
			else
				throw Exception("max_iter should be of type `int`.");
		

		PyObject* threshold = PyDict_GetItemString(parameters, "threshold");
		if(threshold)
			if(PyFloat_Check(threshold))
				params.threshold = PyFloat_AsDouble(threshold);
			else if(PyInt_Check(threshold))
				params.threshold = static_cast<double>(PyFloat_AsDouble(threshold));
			else
				throw Exception("threshold should be of type `float`.");

		PyObject* num_grad = PyDict_GetItemString(parameters, "num_grad");
		if(num_grad)
			if(PyInt_Check(num_grad))
				params.numGrad = PyInt_AsLong(num_grad);
			else if(PyFloat_Check(num_grad))
				params.numGrad = static_cast<int>(PyFloat_AsDouble(num_grad));
			else
				throw Exception("num_grad should be of type `int`.");

		PyObject* batch_size = PyDict_GetItemString(parameters, "batch_size");
		if(batch_size)
			if(PyInt_Check(batch_size))
				params.batchSize = PyInt_AsLong(batch_size);
			else if(PyFloat_Check(batch_size))
				params.batchSize = static_cast<int>(PyFloat_AsDouble(batch_size));
			else
				throw Exception("batch_size should be of type `int`.");

		PyObject* callback = PyDict_GetItemString(parameters, "callback");
		if(callback)
			if(PyCallable_Check(callback))
				params.callback = new CallbackTrain(self, callback);
			else if(callback != Py_None)
				throw Exception("callback should be a function or callable object.");

		PyObject* cb_iter = PyDict_GetItemString(parameters, "cb_iter");
		if(cb_iter)
			if(PyInt_Check(cb_iter))
				params.cbIter = PyInt_AsLong(cb_iter);
			else if(PyFloat_Check(cb_iter))
				params.cbIter = static_cast<int>(PyFloat_AsDouble(cb_iter));
			else
				throw Exception("cb_iter should be of type `int`.");

		PyObject* train_priors = PyDict_GetItemString(parameters, "train_priors");
		if(train_priors)
			if(PyBool_Check(train_priors))
				params.trainPriors = (train_priors == Py_True);
			else
				throw Exception("train_priors should be of type `bool`.");

		PyObject* train_scales = PyDict_GetItemString(parameters, "train_scales");
		if(train_scales)
			if(PyBool_Check(train_scales))
				params.trainScales = (train_scales == Py_True);
			else
				throw Exception("train_scales should be of type `bool`.");

		PyObject* train_weights = PyDict_GetItemString(parameters, "train_weights");
		if(train_weights)
			if(PyBool_Check(train_weights))
				params.trainWeights = (train_weights == Py_True);
			else
				throw Exception("train_weights should be of type `bool`.");

		PyObject* train_features = PyDict_GetItemString(parameters, "train_features");
		if(train_features)
			if(PyBool_Check(train_features))
				params.trainFeatures = (train_features == Py_True);
			else
				throw Exception("train_features should be of type `bool`.");

		PyObject* train_cholesky_factors = PyDict_GetItemString(parameters, "train_cholesky_factors");
		if(train_cholesky_factors)
			if(PyBool_Check(train_cholesky_factors))
				params.trainCholeskyFactors = (train_cholesky_factors == Py_True);
			else
				throw Exception("train_cholesky_factors should be of type `bool`.");

		PyObject* train_predictors = PyDict_GetItemString(parameters, "train_predictors");
		if(train_predictors)
			if(PyBool_Check(train_predictors))
				params.trainPredictors = (train_predictors == Py_True);
			else
				throw Exception("train_predictors should be of type `bool`.");
	}

	return params;
}


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
	try {
		self->mcgsm = new MCGSM(dim_in, dim_out, num_components, num_scales, num_features);
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

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



//PyObject* MCGSM_normalize(MCGSMObject* self, PyObject*, PyObject*) {
//	try {
//		self->mcgsm->normalize();
//	} catch(Exception exception) {
//		PyErr_SetString(PyExc_RuntimeError, exception.message());
//		return 0;
//	}
//
//	Py_INCREF(Py_None);
//	return Py_None;
//}



PyObject* MCGSM_train(MCGSMObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"input", "output", "parameters", 0};

	PyObject* input;
	PyObject* output;
	PyObject* parameters = 0;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "OO|O", const_cast<char**>(kwlist),
		&input,
		&output,
		&parameters))
		return 0;

	// make sure data is stored in NumPy array
	input = PyArray_FROM_OTF(input, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
	output = PyArray_FROM_OTF(output, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

	if(!input || !output) {
		PyErr_SetString(PyExc_TypeError, "Data has to be stored in NumPy arrays.");
		return 0;
	}

	try {
		if(self->mcgsm->train(
				PyArray_ToMatrixXd(input), 
				PyArray_ToMatrixXd(output), 
				PyObject_ToParameters(self, parameters)))
		{
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



PyObject* MCGSM_check_performance(MCGSMObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"input", "output", "repetitions", "parameters", 0};

	PyObject* input;
	PyObject* output;
	int repetitions = 2;
	PyObject* parameters = 0;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "OO|iO", const_cast<char**>(kwlist),
		&input,
		&output,
		&repetitions,
		&parameters))
		return 0;

	// make sure data is stored in NumPy array
	input = PyArray_FROM_OTF(input, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
	output = PyArray_FROM_OTF(output, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

	if(!input || !output) {
		PyErr_SetString(PyExc_TypeError, "Data has to be stored in NumPy arrays.");
		return 0;
	}

	try {
		double err = self->mcgsm->checkPerformance(
			PyArray_ToMatrixXd(input),
			PyArray_ToMatrixXd(output),
			repetitions, 
			PyObject_ToParameters(self, parameters));
		Py_DECREF(input);
		Py_DECREF(output);
		return PyFloat_FromDouble(err);
	} catch(Exception exception) {
		Py_DECREF(input);
		Py_DECREF(output);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	return 0;
}



PyObject* MCGSM_check_gradient(MCGSMObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"input", "output", "epsilon", "parameters", 0};

	PyObject* input;
	PyObject* output;
	double epsilon = 1e-5;
	PyObject* parameters = 0;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "OO|dO", const_cast<char**>(kwlist),
		&input,
		&output,
		&epsilon,
		&parameters))
		return 0;

	// make sure data is stored in NumPy array
	input = PyArray_FROM_OTF(input, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
	output = PyArray_FROM_OTF(output, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

	if(!input || !output) {
		PyErr_SetString(PyExc_TypeError, "Data has to be stored in NumPy arrays.");
		return 0;
	}

	try {
		double err = self->mcgsm->checkGradient(
			PyArray_ToMatrixXd(input),
			PyArray_ToMatrixXd(output),
			epsilon,
			PyObject_ToParameters(self, parameters));
		Py_DECREF(input);
		Py_DECREF(output);
		return PyFloat_FromDouble(err);
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

	return 0;
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

	return 0;
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
		PyObject* result = PyArray_FromMatrixXi(
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

	return 0;
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

	return 0;
}



PyObject* MCGSM_parameters(MCGSMObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"parameters", 0};

	PyObject* parameters = 0;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "|O", const_cast<char**>(kwlist), &parameters))
		return 0;

	try {
		MCGSM::Parameters params = PyObject_ToParameters(self, parameters);

		lbfgsfloatval_t* x = self->mcgsm->parameters(params);

		PyObject* xObj = PyArray_FromMatrixXd(
			Map<Matrix<lbfgsfloatval_t, Dynamic, Dynamic> >(
				x, self->mcgsm->numParameters(params), 1));

		lbfgs_free(x);

		return xObj;
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	return 0;
}



PyObject* MCGSM_set_parameters(MCGSMObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"x", "parameters", 0};

	PyObject* x;
	PyObject* parameters = 0;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O|O", const_cast<char**>(kwlist),
		&x,
		&parameters))
		return 0;

	x = PyArray_FROM_OTF(x, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

	if(!x) {
		PyErr_SetString(PyExc_TypeError, "Parameters have to be stored in NumPy arrays.");
		return 0;
	}

	try {
		self->mcgsm->setParameters(
			PyArray_ToMatrixXd(x).data(), // TODO: PyArray_ToMatrixXd unnecessary
			PyObject_ToParameters(self, parameters));

		Py_DECREF(x);
		Py_INCREF(Py_None);

		return Py_None;
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		Py_DECREF(x);
		return 0;
	}

	return 0;
}



PyObject* MCGSM_compute_gradient(MCGSMObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"input", "output", "x", "parameters", 0};

	PyObject* input;
	PyObject* output;
	PyObject* x;
	PyObject* parameters = 0;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "OOO|O", const_cast<char**>(kwlist),
		&input,
		&output,
		&x,
		&parameters))
		return 0;

	// make sure data is stored in NumPy array
	input = PyArray_FROM_OTF(input, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
	output = PyArray_FROM_OTF(output, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
	x = PyArray_FROM_OTF(x, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

	if(!input || !output || !x) {
		PyErr_SetString(PyExc_TypeError, "Data and parameters have to be stored in NumPy arrays.");
		return 0;
	}

	try {
		MCGSM::Parameters params = PyObject_ToParameters(self, parameters);

		MatrixXd gradient(self->mcgsm->numParameters(params), 1);

		self->mcgsm->computeGradient(
			PyArray_ToMatrixXd(input), 
			PyArray_ToMatrixXd(output), 
			PyArray_ToMatrixXd(x).data(), // TODO: PyArray_ToMatrixXd unnecessary
			gradient.data(), // TODO: don't use MatrixXd
			params);

		Py_DECREF(input);
		Py_DECREF(output);
		Py_DECREF(x);

		return PyArray_FromMatrixXd(gradient);
	} catch(Exception exception) {
		Py_DECREF(input);
		Py_DECREF(output);
		Py_DECREF(x);
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
