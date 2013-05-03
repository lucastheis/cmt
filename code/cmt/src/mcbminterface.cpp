#include "conditionaldistributioninterface.h"
#include "exception.h"
#include "callbackinterface.h"
#include "preconditionerinterface.h"

#include "Eigen/Core"
using Eigen::Map;

#include "mcbminterface.h"
using CMT::MCBM;
using CMT::PatchModel;

MCBM::Parameters PyObject_ToMCBMParameters(PyObject* parameters) {
	MCBM::Parameters params;

	// read parameters from dictionary
	if(parameters && parameters != Py_None) {
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
				params.callback = new CallbackInterface(&MCBM_type, callback);
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

		PyObject* val_iter = PyDict_GetItemString(parameters, "val_iter");
		if(val_iter)
			if(PyInt_Check(val_iter))
				params.valIter = PyInt_AsLong(val_iter);
			else if(PyFloat_Check(val_iter))
				params.valIter = static_cast<int>(PyFloat_AsDouble(val_iter));
			else
				throw Exception("val_iter should be of type `int`.");

		PyObject* val_look_ahead = PyDict_GetItemString(parameters, "val_look_ahead");
		if(val_look_ahead)
			if(PyInt_Check(val_look_ahead))
				params.valLookAhead = PyInt_AsLong(val_look_ahead);
			else if(PyFloat_Check(val_look_ahead))
				params.valLookAhead = static_cast<int>(PyFloat_AsDouble(val_look_ahead));
			else
				throw Exception("val_look_ahead should be of type `int`.");

		PyObject* train_priors = PyDict_GetItemString(parameters, "train_priors");
		if(train_priors)
			if(PyBool_Check(train_priors))
				params.trainPriors = (train_priors == Py_True);
			else
				throw Exception("train_priors should be of type `bool`.");

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

		PyObject* train_predictors = PyDict_GetItemString(parameters, "train_predictors");
		if(train_predictors)
			if(PyBool_Check(train_predictors))
				params.trainPredictors = (train_predictors == Py_True);
			else
				throw Exception("train_predictors should be of type `bool`.");

		PyObject* train_input_bias = PyDict_GetItemString(parameters, "train_input_bias");
		if(train_input_bias)
			if(PyBool_Check(train_input_bias))
				params.trainInputBias = (train_input_bias == Py_True);
			else
				throw Exception("train_input_bias should be of type `bool`.");

		PyObject* train_output_bias = PyDict_GetItemString(parameters, "train_output_bias");
		if(train_output_bias)
			if(PyBool_Check(train_output_bias))
				params.trainOutputBias = (train_output_bias == Py_True);
			else
				throw Exception("train_output_bias should be of type `bool`.");

		PyObject* regularize_features = PyDict_GetItemString(parameters, "regularize_features");
		if(regularize_features)
			if(PyFloat_Check(regularize_features))
				params.regularizeFeatures = PyFloat_AsDouble(regularize_features);
			else if(PyInt_Check(regularize_features))
				params.regularizeFeatures = static_cast<double>(PyFloat_AsDouble(regularize_features));
			else
				throw Exception("regularize_features should be of type `float`.");

		PyObject* regularize_predictors = PyDict_GetItemString(parameters, "regularize_predictors");
		if(regularize_predictors)
			if(PyFloat_Check(regularize_predictors))
				params.regularizePredictors = PyFloat_AsDouble(regularize_predictors);
			else if(PyInt_Check(regularize_predictors))
				params.regularizePredictors = static_cast<double>(PyFloat_AsDouble(regularize_predictors));
			else
				throw Exception("regularize_predictors should be of type `float`.");

		PyObject* regularize_weights = PyDict_GetItemString(parameters, "regularize_weights");
		if(regularize_weights)
			if(PyFloat_Check(regularize_weights))
				params.regularizeWeights = PyFloat_AsDouble(regularize_weights);
			else if(PyInt_Check(regularize_weights))
				params.regularizeWeights = static_cast<double>(PyFloat_AsDouble(regularize_weights));
			else
				throw Exception("regularize_weights should be of type `float`.");

		PyObject* regularizer = PyDict_GetItemString(parameters, "regularizer");
		if(regularizer)
			if(PyString_Check(regularizer)) {
				if(PyString_Size(regularizer) != 2)
					throw Exception("Regularizer should be 'L1' or 'L2'.");

				if(PyString_AsString(regularizer)[1] == '1')
					params.regularizer = MCBM::Parameters::L1;
				else
					params.regularizer = MCBM::Parameters::L2;
			} else {
				throw Exception("regularizer should be of type `str`.");
			}
	}

	return params;
}



const char* MCBM_doc =
	"An implementation of a mixture of conditional Boltzmann machines.\n"
	"\n"
	"The distribution defined by the model is\n"
	"\n"
	"$$p(\\mathbf{y} \\mid \\mathbf{x}) \\propto \\sum_{c} \\exp\\left(\\eta_c + \\sum_i \\beta_{ci} \\left(\\mathbf{b}_i^\\top \\mathbf{x}\\right)^2 + \\mathbf{w}_c^\\top \\mathbf{x} + \\mathbf{y}_c^\\top \\mathbf{A}_c \\mathbf{x} + v_c y\\right),$$\n"
	"\n"
	"where $\\mathbf{x} \\in \\{0, 1\\}^N$ and $y \\in \\{0, 1\\}$.\n"
	"\n"
	"To create an MCBM with $N$-dimensional inputs and, for example, 8 components and 100 features $\\mathbf{b}_i$, use\n"
	"\n"
	"\t>>> mcbm = MCBM(N, 8, 100)\n"
	"\n"
	"To access the different parameters, you can use\n"
	"\n"
	"\t>>> mcbm.priors\n"
	"\t>>> mcbm.weights\n"
	"\t>>> mcbm.features\n"
	"\t>>> mcbm.predictors\n"
	"\t>>> mcbm.input_bias\n"
	"\t>>> mcbm.output_bias\n"
	"\n"
	"which correspond to $\\eta_{c}$, $\\beta_{ci}$, $\\mathbf{b}_i$, $\\mathbf{A}_c$, $\\mathbf{w}_c$,"
	"and $v_c$, respectively.\n"
	"\n"
	"@type  dim_in: integer\n"
	"@param dim_in: dimensionality of input\n"
	"\n"
	"@type  num_components: integer\n"
	"@param num_components: number of components\n"
	"\n"
	"@type  num_features: integer\n"
	"@param num_features: number of features used to approximate input covariance matrices";

int MCBM_init(MCBMObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"dim_in", "num_components", "num_features", 0};

	int dim_in;
	int num_components = 8;
	int num_features = 0;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "i|ii", const_cast<char**>(kwlist),
		&dim_in, &num_components, &num_features))
		return -1;

	if(!num_features)
		num_features = dim_in;

	// create actual MCBM instance
	try {
		self->mcbm = new MCBM(dim_in, num_components, num_features);
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	return 0;
}



PyObject* MCBM_dim_in(MCBMObject* self, void*) {
	return PyInt_FromLong(self->mcbm->dimIn());
}



PyObject* MCBM_dim_out(MCBMObject* self, void*) {
	return PyInt_FromLong(self->mcbm->dimOut());
}



PyObject* MCBM_num_components(MCBMObject* self, void*) {
	return PyInt_FromLong(self->mcbm->numComponents());
}



PyObject* MCBM_num_features(MCBMObject* self, void*) {
	return PyInt_FromLong(self->mcbm->numFeatures());
}



PyObject* MCBM_priors(MCBMObject* self, void*) {
	PyObject* array = PyArray_FromMatrixXd(self->mcbm->priors());

	// make array immutable
	reinterpret_cast<PyArrayObject*>(array)->flags &= ~NPY_WRITEABLE;

	return array;
}



int MCBM_set_priors(MCBMObject* self, PyObject* value, void*) {
	value = PyArray_FROM_OTF(value, NPY_DOUBLE, NPY_IN_ARRAY);

	if(!value) {
		PyErr_SetString(PyExc_TypeError, "Priors should be of type `ndarray`.");
		return -1;
	}

	try {
		self->mcbm->setPriors(PyArray_ToMatrixXd(value));
	} catch(Exception exception) {
		Py_DECREF(value);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	Py_DECREF(value);

	return 0;
}



PyObject* MCBM_weights(MCBMObject* self, void*) {
	PyObject* array = PyArray_FromMatrixXd(self->mcbm->weights());

	// make array immutable
	reinterpret_cast<PyArrayObject*>(array)->flags &= ~NPY_WRITEABLE;

	return array;
}



int MCBM_set_weights(MCBMObject* self, PyObject* value, void*) {
	value = PyArray_FROM_OTF(value, NPY_DOUBLE, NPY_IN_ARRAY);

	if(!value) {
		PyErr_SetString(PyExc_TypeError, "Weights should be of type `ndarray`.");
		return -1;
	}

	try {
		self->mcbm->setWeights(PyArray_ToMatrixXd(value));
	} catch(Exception exception) {
		Py_DECREF(value);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	Py_DECREF(value);

	return 0;
}



PyObject* MCBM_features(MCBMObject* self, void*) {
	PyObject* array = PyArray_FromMatrixXd(self->mcbm->features());

	// make array immutable
	reinterpret_cast<PyArrayObject*>(array)->flags &= ~NPY_WRITEABLE;

	return array;
}



int MCBM_set_features(MCBMObject* self, PyObject* value, void*) {
	value = PyArray_FROM_OTF(value, NPY_DOUBLE, NPY_IN_ARRAY);

	if(!value) {
		PyErr_SetString(PyExc_TypeError, "Features should be of type `ndarray`.");
		return -1;
	}

	try {
		self->mcbm->setFeatures(PyArray_ToMatrixXd(value));
	} catch(Exception exception) {
		Py_DECREF(value);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	Py_DECREF(value);

	return 0;
}



PyObject* MCBM_predictors(MCBMObject* self, void*) {
	PyObject* array = PyArray_FromMatrixXd(self->mcbm->predictors());

	// make array immutable
	reinterpret_cast<PyArrayObject*>(array)->flags &= ~NPY_WRITEABLE;

	return array;
}



int MCBM_set_predictors(MCBMObject* self, PyObject* value, void*) {
	value = PyArray_FROM_OTF(value, NPY_DOUBLE, NPY_IN_ARRAY);

	if(!value) {
		PyErr_SetString(PyExc_TypeError, "Predictors should be of type `ndarray`.");
		return -1;
	}

	try {
		self->mcbm->setPredictors(PyArray_ToMatrixXd(value));
	} catch(Exception exception) {
		Py_DECREF(value);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	Py_DECREF(value);

	return 0;
}



PyObject* MCBM_input_bias(MCBMObject* self, void*) {
	PyObject* array = PyArray_FromMatrixXd(self->mcbm->inputBias());

	// make array immutable
	reinterpret_cast<PyArrayObject*>(array)->flags &= ~NPY_WRITEABLE;

	return array;
}



int MCBM_set_input_bias(MCBMObject* self, PyObject* value, void*) {
	value = PyArray_FROM_OTF(value, NPY_DOUBLE, NPY_IN_ARRAY);

	if(!value) {
		PyErr_SetString(PyExc_TypeError, "Bias vectors should be of type `ndarray`.");
		return -1;
	}

	try {
		self->mcbm->setInputBias(PyArray_ToMatrixXd(value));
	} catch(Exception exception) {
		Py_DECREF(value);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	Py_DECREF(value);

	return 0;
}



PyObject* MCBM_output_bias(MCBMObject* self, void*) {
	PyObject* array = PyArray_FromMatrixXd(self->mcbm->outputBias());

	// make array immutable
	reinterpret_cast<PyArrayObject*>(array)->flags &= ~NPY_WRITEABLE;

	return array;
}



int MCBM_set_output_bias(MCBMObject* self, PyObject* value, void*) {
	value = PyArray_FROM_OTF(value, NPY_DOUBLE, NPY_IN_ARRAY);

	if(!value) {
		PyErr_SetString(PyExc_TypeError, "Bias vectors should be of type `ndarray`.");
		return -1;
	}

	try {
		self->mcbm->setOutputBias(PyArray_ToMatrixXd(value));
	} catch(Exception exception) {
		Py_DECREF(value);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	Py_DECREF(value);

	return 0;
}



const char* MCBM_train_doc =
	"train(self, input, output, input_val=None, output_val=None, parameters=None)\n"
	"\n"
	"Fits model parameters to given data using L-BFGS.\n"
	"\n"
	"The following example demonstrates possible parameters and default settings.\n"
	"\n"
	"\t>>> model.train(input, output, parameters={\n"
	"\t>>> \t'verbosity': 0,\n"
	"\t>>> \t'max_iter': 1000,\n"
	"\t>>> \t'threshold': 1e-5,\n"
	"\t>>> \t'num_grad': 20,\n"
	"\t>>> \t'batch_size': 2000,\n"
	"\t>>> \t'callback': None,\n"
	"\t>>> \t'cb_iter': 25,\n"
	"\t>>> \t'val_iter': 25,\n"
	"\t>>> \t'train_priors': True,\n"
	"\t>>> \t'train_weights': True,\n"
	"\t>>> \t'train_features': True,\n"
	"\t>>> \t'train_predictors': True,\n"
	"\t>>> \t'train_input_bias': True,\n"
	"\t>>> \t'train_output_bias': True,\n"
	"\t>>> \t'regularizer': 'L1',\n"
	"\t>>> \t'regularize_features': 0.,\n"
	"\t>>> \t'regularize_weights': 0.,\n"
	"\t>>> \t'regularize_predictors': 0.\n"
	"\t>>> })\n"
	"\n"
	"The parameters C{train_priors}, C{train_weights}, and so on can be used to control which "
	"parameters will be optimized. Optimization stops after C{max_iter} iterations or if "
	"the gradient is sufficiently small enough, as specified by C{threshold}."
	"C{num_grad} is the number of gradients used by L-BFGS to approximate the inverse Hessian "
	"matrix.\n"
	"\n"
	"The parameter C{batch_size} has no effect on the solution of the optimization but "
	"can affect speed by reducing the number of cache misses.\n"
	"\n"
	"If a callback function is given, it will be called every C{cb_iter} iterations. The first "
	"argument to callback will be the current iteration, the second argument will be a I{copy} of "
	"the model.\n"
	"\n"
	"\t>>> def callback(i, mcbm):\n"
	"\t>>> \tprint i\n"
	"\n"
	"@type  input: ndarray\n"
	"@param input: inputs stored in columns\n"
	"\n"
	"@type  output: ndarray\n"
	"@param output: outputs stored in columns\n"
	"\n"
	"@type  input_val: ndarray\n"
	"@param input_val: inputs used for early stopping based on validation error\n"
	"\n"
	"@type  output_val: ndarray\n"
	"@param output_val: outputs used for early stopping based on validation error\n"
	"\n"
	"@type  parameters: dict\n"
	"@param parameters: a dictionary containing hyperparameters\n"
	"\n"
	"@rtype: bool\n"
	"@return: C{True} if training converged, otherwise C{False}";

PyObject* MCBM_train(MCBMObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"input", "output", "input_val", "output_val", "parameters", 0};

	PyObject* input;
	PyObject* output;
	PyObject* input_val = 0;
	PyObject* output_val = 0;
	PyObject* parameters = 0;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "OO|OOO", const_cast<char**>(kwlist),
		&input,
		&output,
		&input_val,
		&output_val,
		&parameters))
		return 0;

	// make sure data is stored in NumPy array
	input = PyArray_FROM_OTF(input, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
	output = PyArray_FROM_OTF(output, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

	if(!input || !output) {
		Py_XDECREF(input);
		Py_XDECREF(output);
		PyErr_SetString(PyExc_TypeError, "Data has to be stored in NumPy arrays.");
		return 0;
	}

	if(input_val == Py_None)
		input_val = 0;
	if(output_val == Py_None)
		output_val = 0;

	if(input_val || output_val) {
		input_val = PyArray_FROM_OTF(input_val, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
		output_val = PyArray_FROM_OTF(output_val, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

		if(!input_val || !output_val) {
			Py_DECREF(input);
			Py_DECREF(output);
			Py_XDECREF(input_val);
			Py_XDECREF(output_val);
			PyErr_SetString(PyExc_TypeError, "Validation data has to be stored in NumPy arrays.");
			return 0;
		}
	}

	try {
		bool converged;

		if(input_val && output_val) {
			converged = self->mcbm->train(
				PyArray_ToMatrixXd(input), 
				PyArray_ToMatrixXd(output), 
				PyArray_ToMatrixXd(input_val),
				PyArray_ToMatrixXd(output_val),
				PyObject_ToMCBMParameters(parameters));
		} else {
			converged = self->mcbm->train(
				PyArray_ToMatrixXd(input), 
				PyArray_ToMatrixXd(output), 
				PyObject_ToMCBMParameters(parameters));
		}

		if(converged) {
			Py_DECREF(input);
			Py_DECREF(output);
			Py_XDECREF(input_val);
			Py_XDECREF(output_val);
			Py_INCREF(Py_True);
			return Py_True;
		} else {
			Py_DECREF(input);
			Py_DECREF(output);
			Py_XDECREF(input_val);
			Py_XDECREF(output_val);
			Py_INCREF(Py_False);
			return Py_False;
		}
	} catch(Exception exception) {
		Py_DECREF(input);
		Py_DECREF(output);
		Py_XDECREF(input_val);
		Py_XDECREF(output_val);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	return 0;
}



const char* MCBM_parameters_doc =
	"parameters(self, parameters=None)\n"
	"\n"
	"Summarizes the parameters of the model in a long vector.\n"
	"\n"
	"If C{parameters} is given, only the parameters with C{train_* = True} will be contained "
	"in the vector.\n"
	"\n"
	"@type  parameters: dict\n"
	"@param parameters: a dictionary containing hyperparameters\n"
	"\n"
	"@rtype: ndarray\n"
	"@return: model parameters vectorized and concatenated";

PyObject* MCBM_parameters(MCBMObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"parameters", 0};

	PyObject* parameters = 0;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "|O", const_cast<char**>(kwlist), &parameters))
		return 0;

	try {
		MCBM::Parameters params = PyObject_ToMCBMParameters(parameters);

		lbfgsfloatval_t* x = self->mcbm->parameters(params);

		PyObject* xObj = PyArray_FromMatrixXd(
			Map<Matrix<lbfgsfloatval_t, Dynamic, Dynamic> >(
				x, self->mcbm->numParameters(params), 1));

		lbfgs_free(x);

		return xObj;
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	return 0;
}



const char* MCBM_set_parameters_doc =
	"set_parameters(self, x, parameters=None)\n"
	"\n"
	"Loads all model parameters from a vector as produced by L{parameters()}.\n"
	"\n"
	"@type  x: ndarray\n"
	"@param x: all model parameters concatenated to a vector\n"
	"\n"
	"@type  parameters: dict\n"
	"@param parameters: a dictionary containing hyperparameters";

PyObject* MCBM_set_parameters(MCBMObject* self, PyObject* args, PyObject* kwds) {
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
		self->mcbm->setParameters(
			PyArray_ToMatrixXd(x).data(), // TODO: PyArray_ToMatrixXd unnecessary
			PyObject_ToMCBMParameters(parameters));

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



PyObject* MCBM_parameter_gradient(MCBMObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"input", "output", "x", "parameters", 0};

	PyObject* input;
	PyObject* output;
	PyObject* x = 0;
	PyObject* parameters = 0;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "OO|OO", const_cast<char**>(kwlist),
		&input,
		&output,
		&x,
		&parameters))
		return 0;

	// make sure data is stored in NumPy array
	input = PyArray_FROM_OTF(input, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
	output = PyArray_FROM_OTF(output, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

	if(!input || !output) {
		Py_XDECREF(input);
		Py_XDECREF(output);
		PyErr_SetString(PyExc_TypeError, "Data has to be stored in NumPy arrays.");
		return 0;
	}

	if(x)
		x = PyArray_FROM_OTF(x, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

	try {
		MCBM::Parameters params = PyObject_ToMCBMParameters(parameters);

		MatrixXd gradient(self->mcbm->numParameters(params), 1); // TODO: don't use MatrixXd

		if(x) {
			#if LBFGS_FLOAT == 64
			self->mcbm->parameterGradient(
				PyArray_ToMatrixXd(input),
				PyArray_ToMatrixXd(output),
				reinterpret_cast<lbfgsfloatval_t*>(PyArray_DATA(x)),
				gradient.data(),
				params);
			#elif LBFGS_FLOAT == 32
			lbfgsfloatval_t* xLBFGS = lbfgs_malloc(PyArray_SIZE(x));
			lbfgsfloatval_t* gLBFGS = lbfgs_malloc(PyArray_SIZE(x));
			double* xData = reinterpret_cast<double*>(PyArray_DATA(x));

			for(int i = 0; i < PyArray_SIZE(x); ++i)
				xLBFGS[i] = static_cast<lbfgsfloatval_t>(xData[i]);

			self->mcbm->parameterGradient(
				PyArray_ToMatrixXd(input),
				PyArray_ToMatrixXd(output),
				xLBFGS,
				gLBFGS,
				params);

			for(int i = 0; i < PyArray_SIZE(x); ++i)
				gradient.data()[i] = static_cast<double>(gLBFGS[i]);

			lbfgs_free(gLBFGS);
			lbfgs_free(xLBFGS);
			#else
			#error "LibLBFGS is configured in a way I don't understand."
			#endif
		} else {
			lbfgsfloatval_t* x = self->mcbm->parameters(params);

			self->mcbm->parameterGradient(
				PyArray_ToMatrixXd(input),
				PyArray_ToMatrixXd(output),
				x,
				gradient.data(),
				params);

			lbfgs_free(x);
		}

		Py_DECREF(input);
		Py_DECREF(output);
		Py_XDECREF(x);

		return PyArray_FromMatrixXd(gradient);
	} catch(Exception exception) {
		Py_DECREF(input);
		Py_DECREF(output);
		Py_XDECREF(x);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}
}



PyObject* MCBM_check_gradient(MCBMObject* self, PyObject* args, PyObject* kwds) {
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
		Py_XDECREF(input);
		Py_XDECREF(output);
		PyErr_SetString(PyExc_TypeError, "Data has to be stored in NumPy arrays.");
		return 0;
	}

	try {
		double err = self->mcbm->checkGradient(
			PyArray_ToMatrixXd(input),
			PyArray_ToMatrixXd(output),
			epsilon,
			PyObject_ToMCBMParameters(parameters));
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



const char* MCBM_reduce_doc =
	"__reduce__(self)\n"
	"\n"
	"Method used by Pickle.";

PyObject* MCBM_reduce(MCBMObject* self, PyObject*) {
	// constructor arguments
	PyObject* args = Py_BuildValue("(iii)", 
		self->mcbm->dimIn(),
		self->mcbm->numComponents(),
		self->mcbm->numFeatures());

	// parameters
	PyObject* priors = MCBM_priors(self, 0);
	PyObject* weights = MCBM_weights(self, 0);
	PyObject* features = MCBM_features(self, 0);
	PyObject* predictors = MCBM_predictors(self, 0);
	PyObject* input_bias = MCBM_input_bias(self, 0);
	PyObject* output_bias = MCBM_output_bias(self, 0);

	PyObject* state = Py_BuildValue("(OOOOOO)", 
		priors, weights, features, predictors, input_bias, output_bias);

	Py_DECREF(priors);
	Py_DECREF(weights);
	Py_DECREF(features);
	Py_DECREF(predictors);
	Py_DECREF(input_bias);
	Py_DECREF(output_bias);

	PyObject* result = Py_BuildValue("(OOO)", Py_TYPE(self), args, state);

	Py_DECREF(args);
	Py_DECREF(state);

	return result;
}



const char* MCBM_setstate_doc =
	"__setstate__(self)\n"
	"\n"
	"Method used by Pickle.";

PyObject* MCBM_setstate(MCBMObject* self, PyObject* state) {
	PyObject* priors;
	PyObject* weights;
	PyObject* features;
	PyObject* predictors;
	PyObject* input_bias;
	PyObject* output_bias;

	if(!PyArg_ParseTuple(state, "(OOOOOO)",
		&priors, &weights, &features, &predictors, &input_bias, &output_bias))
		return 0;

	try {
		MCBM_set_priors(self, priors, 0);
		MCBM_set_weights(self, weights, 0);
		MCBM_set_features(self, features, 0);
		MCBM_set_predictors(self, predictors, 0);
		MCBM_set_input_bias(self, input_bias, 0);
		MCBM_set_output_bias(self, output_bias, 0);
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	Py_INCREF(Py_None);
	return Py_None;
}



const char* PatchMCBM_doc =
	"Model image patches by using an L{MCBM} for each conditional distribution.\n"
	"\n"
	"@type  rows: integer\n"
	"@param rows: number of rows of the image patch\n"
	"\n"
	"@type  cols: integer\n"
	"@param cols: number of columns of the image patch\n"
	"\n"
	"@type  xmask: C{ndarray}\n"
	"@param xmask: a Boolean array describing the input pixels\n"
	"\n"
	"@type  ymask: C{ndarray}\n"
	"@param ymask: a Boolean array describing the output pixels\n"
	"\n"
	"@type  model: L{MCBM}\n"
	"@param model: model used as a template to initialize all conditional distributions\n"
	"\n"
	"@type  max_pcs: integer\n"
	"@param max_pcs: can be used to reduce dimensionality of inputs to conditional models";

int PatchMCBM_init(PatchMCBMObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"rows", "cols", "xmask", "ymask", "model", "max_pcs", 0};

	int rows;
	int cols;
	PyObject* xmask;
	PyObject* ymask;
	PyObject* model = 0;
	int max_pcs = -1;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "iiOO|Oi", const_cast<char**>(kwlist),
		&rows, &cols, &xmask, &ymask, &model, &max_pcs))
		return -1;

	if(model == Py_None)
		model = 0;

	if(model && !PyType_IsSubtype(Py_TYPE(model), &MCBM_type)) {
		PyErr_SetString(PyExc_TypeError, "Model has to be of type `MCBM`.");
		return 0;
	}

	xmask = PyArray_FROM_OTF(xmask, NPY_BOOL, NPY_F_CONTIGUOUS | NPY_ALIGNED);
	ymask = PyArray_FROM_OTF(ymask, NPY_BOOL, NPY_F_CONTIGUOUS | NPY_ALIGNED);

	if(!xmask || !ymask) {
		Py_XDECREF(xmask);
		Py_XDECREF(ymask);
		PyErr_SetString(PyExc_TypeError, "Masks have to be given as Boolean arrays.");
		return 0;
	}

	// create the actual model
	try {
		self->patchMCBM = new PatchModel<MCBM, CMT::PCATransform>(
			rows,
			cols,
			PyArray_ToMatrixXb(xmask),
			PyArray_ToMatrixXb(ymask),
			model ? reinterpret_cast<MCBMObject*>(model)->mcbm : 0,
			max_pcs);
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	return 0;
}



PyObject* PatchMCBM_rows(PatchMCBMObject* self, void*) {
	return PyInt_FromLong(self->patchMCBM->rows());
}



PyObject* PatchMCBM_cols(PatchMCBMObject* self, void*) {
	return PyInt_FromLong(self->patchMCBM->cols());
}



PyObject* PatchMCBM_input_mask(PatchMCBMObject* self, PyObject* args) {
	int i = -1;
	int j = -1;

	if(args && !PyArg_ParseTuple(args, "|ii", &i, &j))
		return 0;

	if(i >= 0 && j < 0) {
		PyErr_SetString(PyExc_TypeError, "Index should consist of a row and a column.");
		return 0;
	}

	PyObject* array;

	if(i < 0 || j < 0)
		array = PyArray_FromMatrixXb(self->patchMCBM->inputMask());
	else
		array = PyArray_FromMatrixXb(self->patchMCBM->inputMask(i, j));

	// make array immutable
	reinterpret_cast<PyArrayObject*>(array)->flags &= ~NPY_WRITEABLE;
	return array;
}



PyObject* PatchMCBM_output_mask(PatchMCBMObject* self, PyObject* args) {
	int i = -1;
	int j = -1;

	if(args && !PyArg_ParseTuple(args, "|ii", &i, &j))
		return 0;

	if(i >= 0 && j < 0) {
		PyErr_SetString(PyExc_TypeError, "Index should consist of a row and a column.");
		return 0;
	}

	PyObject* array;

	if(i < 0 || j < 0)
		array = PyArray_FromMatrixXb(self->patchMCBM->outputMask());
	else
		array = PyArray_FromMatrixXb(self->patchMCBM->outputMask(i, j));

	// make array immutable
	reinterpret_cast<PyArrayObject*>(array)->flags &= ~NPY_WRITEABLE;
	return array;
}



PyObject* PatchMCBM_subscript(PatchMCBMObject* self, PyObject* key) {
	if(!PyTuple_Check(key)) {
		PyErr_SetString(PyExc_TypeError, "Index must be a tuple.");
		return 0;
	}

	int i;
	int j;

	if(!PyArg_ParseTuple(key, "ii", &i, &j)) {
		PyErr_SetString(PyExc_TypeError, "Index should consist of a row and a column.");
		return 0;
	}

	PyObject* mcbmObject = CD_new(&MCBM_type, 0, 0);
	reinterpret_cast<MCBMObject*>(mcbmObject)->mcbm = &self->patchMCBM->operator()(i, j);
	reinterpret_cast<MCBMObject*>(mcbmObject)->owner = false;
	Py_INCREF(mcbmObject);

	return mcbmObject;
}



int PatchMCBM_ass_subscript(PatchMCBMObject* self, PyObject* key, PyObject* value) {
	if(!PyType_IsSubtype(Py_TYPE(value), &MCBM_type)) {
		PyErr_SetString(PyExc_TypeError, "Conditional distribution should be an MCBM.");
		return -1;
	}

	if(!PyTuple_Check(key)) {
		PyErr_SetString(PyExc_TypeError, "Index must be a tuple.");
		return -1;
	}

	int i;
	int j;

	if(!PyArg_ParseTuple(key, "ii", &i, &j)) {
		PyErr_SetString(PyExc_TypeError, "Index should consist of a row and a column.");
		return -1;
	}

 	self->patchMCBM->operator()(i, j) = *reinterpret_cast<MCBMObject*>(value)->mcbm;

	return 0;
}



PyObject* PatchMCBM_preconditioner(PatchMCBMObject* self, PyObject* args) {
	int i;
	int j;

	if(!PyArg_ParseTuple(args, "ii", &i, &j)) {
		PyErr_SetString(PyExc_TypeError, "Index should consist of a row and a column.");
		return 0;
	}

	try {
		PCATransform* pc = &self->patchMCBM->preconditioner(i, j);
		PyObject* preconditioner = Preconditioner_new(&PCATransform_type, 0, 0);
		reinterpret_cast<PCATransformObject*>(preconditioner)->owner = false;
		reinterpret_cast<PCATransformObject*>(preconditioner)->preconditioner = pc;
		Py_INCREF(preconditioner);
		return preconditioner;
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	return 0;
}



PyObject* PatchMCBM_preconditioners(PatchMCBMObject* self, void*) {
	if(self->patchMCBM->maxPCs() < 0)
		return PyDict_New();

	PyObject* preconditioners = PyDict_New();

	for(int i = 0; i < self->patchMCBM->rows(); ++i) {
		for(int j = 0; j < self->patchMCBM->cols(); ++j) {
			PyObject* index = Py_BuildValue("(ii)", i, j);
			PyObject* preconditioner = PatchMCBM_preconditioner(self, index);

			if(!preconditioner) {
				PyErr_Clear();
				Py_DECREF(index);
				continue;
			}

			PyDict_SetItem(preconditioners, index, preconditioner);

			Py_DECREF(index);
			Py_DECREF(preconditioner);
		}
	}

	return preconditioners;
}



int PatchMCBM_set_preconditioners(PatchMCBMObject* self, PyObject* value, void*) {
	if(!PyDict_Check(value)) {
		PyErr_SetString(PyExc_RuntimeError, "Preconditioners have to be stored in a dictionary."); 
		return -1;
	}

	for(int i = 0; i < self->patchMCBM->rows(); ++i)
		for(int j = 0; j < self->patchMCBM->cols(); ++j) {
			PyObject* index = Py_BuildValue("(ii)", i, j);
			PyObject* preconditioner = PyDict_GetItem(value, index);

			if(!preconditioner)
				continue;

			if(!PyType_IsSubtype(Py_TYPE(preconditioner), &PCATransform_type)) {
				PyErr_SetString(PyExc_RuntimeError,
					"All preconditioners must be of type `PCATransform`.");
				return -1;
			}

			try {
 				self->patchMCBM->setPreconditioner(i, j,
 					*reinterpret_cast<PCATransformObject*>(preconditioner)->preconditioner);
			} catch(Exception exception) {
				PyErr_SetString(PyExc_RuntimeError, exception.message());
				return -1;
			}

			Py_DECREF(index);
		}

	return 0;
}



const char* PatchMCBM_initialize_doc =
	"initialize(self, data, parameters=None)\n"
	"\n"
	"Trains the model assuming shift-invariance of the patch statistics.\n"
	"\n"
	"A single conditional distribution is fitted to the given data and all models with\n"
	"a I{complete} neighborhood are initialized with this one set of parameters.\n"
	"\n"
	"It is assumed that the patches are stored in row-order ('C') in the columns of\n"
	"L{data}.\n"
	"\n"
	"@type  data: ndarray\n"
	"@param data: image patches stored column-wise";

PyObject* PatchMCBM_initialize(PatchMCBMObject* self, PyObject* args, PyObject* kwds) {
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
		PyErr_SetString(PyExc_TypeError, "Data has to be stored in NumPy arrays.");
		return 0;
	}

	try {
		self->patchMCBM->initialize(
			PyArray_ToMatrixXd(data),
			PyObject_ToMCBMParameters(parameters));
		Py_DECREF(data);
		Py_INCREF(Py_None);
		return Py_None;
	} catch(Exception exception) {
		Py_DECREF(data);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	return 0;
}



const char* PatchMCBM_train_doc =
	"train(self, data, dat_val=None, parameters=None)\n"
	"\n"
	"Trains the model to the given image patches by fitting each conditional\n"
	"distribution in turn.\n"
	"\n"
	"It is assumed that the patches are stored in row-order ('C') in the columns of\n"
	"L{data}. If hyperparameters are given, they are passed on to each conditional\n"
	"distribution.\n"
	"\n"
	"@type  data: ndarray\n"
	"@param data: image patches stored column-wise\n"
	"\n"
	"@type  data_val: ndarray\n"
	"@param data_val: image patches used for early stopping based on validation error\n"
	"\n"
	"@type  parameters: dict\n"
	"@param parameters: a dictionary containing hyperparameters\n"
	"\n"
	"@rtype: bool\n"
	"@return: C{True} if training of all models converged, otherwise C{False}";

PyObject* PatchMCBM_train(PatchMCBMObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"i", "j", "data", "data_val", "parameters", 0};

	PyObject* data;
	PyObject* data_val = 0;
	PyObject* parameters = 0;
	int i = -1;
	int j = -1;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "iiO|OO", const_cast<char**>(kwlist),
		&i, &j,
		&data,
		&data_val,
		&parameters))
	{
		PyErr_Clear();

		const char* kwlist[] = {"data", "data_val", "parameters", 0};

		if(!PyArg_ParseTupleAndKeywords(args, kwds, "O|OO", const_cast<char**>(kwlist),
			&data,
			&data_val,
			&parameters))
			return 0;
	}

	// make sure data is stored in NumPy array
	data = PyArray_FROM_OTF(data, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

	if(!data) {
		PyErr_SetString(PyExc_TypeError, "Data has to be stored in NumPy array.");
		return 0;
	}

	if(data_val == Py_None)
		data_val = 0;

	if(data_val) {
		data_val = PyArray_FROM_OTF(data_val, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

		if(!data_val) {
			Py_DECREF(data);
			PyErr_SetString(PyExc_TypeError, "Validation data has to be stored in NumPy array.");
			return 0;
		}
	}

	try {
		bool converged;

		if(data_val) {
			if(i > -1 && j > -1)
				converged = self->patchMCBM->train(
					i, j,
					PyArray_ToMatrixXd(data),
					PyArray_ToMatrixXd(data_val),
					PyObject_ToMCBMParameters(parameters));
			else
				converged = self->patchMCBM->train(
					PyArray_ToMatrixXd(data),
					PyArray_ToMatrixXd(data_val),
					PyObject_ToMCBMParameters(parameters));
		} else {
			if(i > -1 && j > -1)
				converged = self->patchMCBM->train(
					i, j,
					PyArray_ToMatrixXd(data),
					PyObject_ToMCBMParameters(parameters));
			else
				converged = self->patchMCBM->train(
					PyArray_ToMatrixXd(data),
					PyObject_ToMCBMParameters(parameters));
		}

		if(converged) {
			Py_DECREF(data);
			Py_XDECREF(data_val);
			Py_INCREF(Py_True);
			return Py_True;
		} else {
			Py_DECREF(data);
			Py_XDECREF(data_val);
			Py_INCREF(Py_False);
			return Py_False;
		}
	} catch(Exception exception) {
		Py_DECREF(data);
		Py_XDECREF(data_val);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	return 0;
}



PyObject* PatchMCBM_loglikelihood(PatchMCBMObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"i", "j", "data", 0};

	PyObject* data;
	int i = -1;
	int j = -1;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "iiO", const_cast<char**>(kwlist),
		&i, &j, &data)) 
	{
		PyErr_Clear();

		const char* kwlist[] = {"data", 0};

		if(!PyArg_ParseTupleAndKeywords(args, kwds, "O", const_cast<char**>(kwlist), &data))
			return 0;
	}

	// make sure data is stored in NumPy array
	data = PyArray_FROM_OTF(data, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

	if(!data) {
		PyErr_SetString(PyExc_TypeError, "Data has to be stored in NumPy arrays.");
		return 0;
	}

	try {
		PyObject* result;
		if(i > -1 && j > -1)
			result = PyArray_FromMatrixXd(
				self->patchMCBM->logLikelihood(i, j, PyArray_ToMatrixXd(data)));
		else
			result = PyArray_FromMatrixXd(
				self->patchMCBM->logLikelihood(PyArray_ToMatrixXd(data)));
		Py_DECREF(data);
		return result;
	} catch(Exception exception) {
		Py_DECREF(data);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	return 0;
}



const char* PatchMCBM_reduce_doc =
	"__reduce__(self)\n"
	"\n"
	"Method used by Pickle.";

PyObject* PatchMCBM_reduce(PatchMCBMObject* self, PyObject*) {
	int rows = self->patchMCBM->rows();
	int cols = self->patchMCBM->cols();
	int maxPCs = self->patchMCBM->maxPCs();

	PyObject* inputMask = PatchMCBM_input_mask(self, 0);
	PyObject* outputMask = PatchMCBM_output_mask(self, 0);

	// constructor arguments
	PyObject* args = Py_BuildValue("(iiOOOi)",
		rows,
		cols,
		inputMask,
		outputMask,
		Py_None,
		maxPCs);
	
	Py_DECREF(inputMask);
	Py_DECREF(outputMask);

	// parameters
	PyObject* models = PyTuple_New(self->patchMCBM->dim());

	for(int i = 0; i < rows; ++i)
		for(int j = 0; j < cols; ++j) {
			PyObject* index = Py_BuildValue("(ii)", i, j);
			PyObject* mcbm = PatchMCBM_subscript(self, index);

			// add MCBM to list of models
			PyTuple_SetItem(models, i * cols + j, mcbm);

			Py_DECREF(index);
		}

	PyObject* preconditioners = PatchMCBM_preconditioners(self, 0);

	PyObject* state = Py_BuildValue("(OO)", models, preconditioners);
	PyObject* result = Py_BuildValue("(OOO)", Py_TYPE(self), args, state);

	Py_DECREF(preconditioners);
	Py_DECREF(args);
	Py_DECREF(state);

	return result;
}



const char* PatchMCBM_setstate_doc =
	"__setstate__(self)\n"
	"\n"
	"Method used by Pickle.";

PyObject* PatchMCBM_setstate(PatchMCBMObject* self, PyObject* state) {
	int cols = self->patchMCBM->cols();

	// for some reason the actual state is encapsulated in another tuple
	state = PyTuple_GetItem(state, 0);

	PyObject* models = PyTuple_GetItem(state, 0);
  	PyObject* preconditioners = PyTuple_GetItem(state, 1);

	if(PyTuple_Size(models) != self->patchMCBM->dim()) {
		PyErr_SetString(PyExc_RuntimeError, "Something went wrong while unpickling the model.");
		return 0;
	}

	try {
		for(int i = 0; i < self->patchMCBM->dim(); ++i) {
			PyObject* index = Py_BuildValue("(ii)", i / cols, i % cols);
			PatchMCBM_ass_subscript(self, index, PyTuple_GetItem(models, i));
			Py_DECREF(index);
		}
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

  	if(preconditioners)
  		PatchMCBM_set_preconditioners(self, preconditioners, 0);

	Py_INCREF(Py_None);
	return Py_None;
}
