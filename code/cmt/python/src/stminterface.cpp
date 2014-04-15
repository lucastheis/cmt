#include "conditionaldistributioninterface.h"
#include "callbackinterface.h"
#include "trainableinterface.h"
#include "stminterface.h"

#include "Eigen/Core"
using Eigen::Map;

#include "cmt/utils"
using CMT::Exception;

Trainable::Parameters* PyObject_ToSTMParameters(PyObject* parameters) {
	STM::Parameters* params = dynamic_cast<STM::Parameters*>(
		PyObject_ToParameters(parameters, new STM::Parameters));

	// read parameters from dictionary
	if(parameters && parameters != Py_None) {
		PyObject* callback = PyDict_GetItemString(parameters, "callback");
		if(callback)
			if(PyCallable_Check(callback))
				params->callback = new CallbackInterface(&STM_type, callback);
			else if(callback != Py_None)
				throw Exception("callback should be a function or callable object.");

		PyObject* train_sharpness = PyDict_GetItemString(parameters, "train_sharpness");
		if(train_sharpness)
			if(PyBool_Check(train_sharpness))
				params->trainSharpness = (train_sharpness == Py_True);
			else
				throw Exception("train_sharpness should be of type `bool`.");

		PyObject* train_biases = PyDict_GetItemString(parameters, "train_biases");
		if(train_biases)
			if(PyBool_Check(train_biases))
				params->trainBiases = (train_biases == Py_True);
			else
				throw Exception("train_biases should be of type `bool`.");

		PyObject* train_weights = PyDict_GetItemString(parameters, "train_weights");
		if(train_weights)
			if(PyBool_Check(train_weights))
				params->trainWeights = (train_weights == Py_True);
			else
				throw Exception("train_weights should be of type `bool`.");

		PyObject* train_features = PyDict_GetItemString(parameters, "train_features");
		if(train_features)
			if(PyBool_Check(train_features))
				params->trainFeatures = (train_features == Py_True);
			else
				throw Exception("train_features should be of type `bool`.");

		PyObject* train_predictors = PyDict_GetItemString(parameters, "train_predictors");
		if(train_predictors)
			if(PyBool_Check(train_predictors))
				params->trainPredictors = (train_predictors == Py_True);
			else
				throw Exception("train_predictors should be of type `bool`.");

		PyObject* train_linear_predictor = PyDict_GetItemString(parameters, "train_linear_predictor");
		if(train_linear_predictor)
			if(PyBool_Check(train_linear_predictor))
				params->trainLinearPredictor = (train_linear_predictor == Py_True);
			else
				throw Exception("train_linear_predictor should be of type `bool`.");

		PyObject* regularize_biases = PyDict_GetItemString(parameters, "regularize_biases");
		if(regularize_biases)
			params->regularizeBiases = PyObject_ToRegularizer(regularize_biases);

		PyObject* regularize_features = PyDict_GetItemString(parameters, "regularize_features");
		if(regularize_features)
			params->regularizeFeatures = PyObject_ToRegularizer(regularize_features);

		PyObject* regularize_predictors = PyDict_GetItemString(parameters, "regularize_predictors");
		if(regularize_predictors)
			params->regularizePredictors = PyObject_ToRegularizer(regularize_predictors);

		PyObject* regularize_weights = PyDict_GetItemString(parameters, "regularize_weights");
		if(regularize_weights)
			params->regularizeWeights = PyObject_ToRegularizer(regularize_weights);

		PyObject* regularize_linear_predictor = PyDict_GetItemString(parameters, "regularize_linear_predictor");
		if(regularize_linear_predictor)
			params->regularizeLinearPredictor = PyObject_ToRegularizer(regularize_linear_predictor);
	}

	return params;
}



const char* STM_doc =
	"An implementation of the spike-triggered mixture model.\n"
	"\n"
	"The conditional distribution defined by the model is\n"
	"\n"
	"$$p(y \\mid \\mathbf{x}, \\mathbf{z}) = q(y \\mid g(f(\\mathbf{x}, \\mathbf{z})))$$\n"
	"\n"
	"where $y$ is a scalar, $\\mathbf{x} \\in \\mathbb{R}^N$, $\\mathbf{z} \\in \\mathbb{R}^M$, $q$\n"
	"is a univariate distribution, $g$ is some nonlinearity, and\n"
	"\n"
	"$$f(\\mathbf{x}, \\mathbf{z}) = \\log \\sum_k \\exp\\left( \\lambda \\left[ \\sum_l \\beta_{kl} (\\mathbf{u}_l^\\top \\mathbf{x})^2 + \\mathbf{w}_k \\mathbf{x} + a_k \\right] \\right) / \\lambda + \\mathbf{v}^\\top \\mathbf{z}.$$\n"
	"\n"
	"As you can see from the equation above, part of the input is processed nonlinearly and part of\n"
	"the input is processed linearly. To create an STM with $N$-dimensional\n"
	"nonlinear inputs $\\mathbf{x}$ and $M$-dimensional linear inputs $\\mathbf{z}$\n"
	"with, for example, 3 components and 20 features $\\mathbf{u}_l$, use\n"
	"\n"
	"\t>>> stm = STM(N, M, 3, 20, LogisticFunction, Bernoulli)\n"
	"\n"
	"To access the different parameters, you can use\n"
	"\n"
	"\t>>> stm.sharpness\n"
	"\t>>> stm.biases\n"
	"\t>>> stm.weights\n"
	"\t>>> stm.features\n"
	"\t>>> stm.predictors\n"
	"\t>>> stm.linear_predictor\n"
	"\t>>> stm.nonlinearity\n"
	"\t>>> stm.distribution\n"
	"\n"
	"which correspond to $a_k$, $\\beta_{kl}$, $\\mathbf{u}_l$, $\\mathbf{w}_k$,\n"
	"$\\mathbf{v}$, $g$, and $q$ respectively.\n"
	"\n"
	"@type  dim_in_nonlinear: C{int}\n"
	"@param dim_in_nonlinear: dimensionality of nonlinear portion of input\n"
	"\n"
	"@type  dim_in_linear: C{int}\n"
	"@param dim_in_linear: dimensionality of linear portion of input (default: 0)\n"
	"\n"
	"@type  num_components: C{int}\n"
	"@param num_components: number of components (default: 3)\n"
	"\n"
	"@type  num_features: C{int}\n"
	"@param num_features: number of quadratic features"
	"\n"
	"@type  nonlinearity: L{Nonlinearity<nonlinear.Nonlinearity>}/C{type}\n"
	"@param nonlinearity: nonlinearity applied to log-sum-exp, $g$ (default: L{LogisticFunction<nonlinear.LogisticFunction>})\n"
	"\n"
	"@type  distribution: L{UnivariateDistribution<models.UnivariateDistribution>}/C{type}\n"
	"@param distribution: distribution of outputs, $q$ (default: L{Bernoulli<models.Bernoulli>})";

int STM_init(STMObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {
		"dim_in_nonlinear",
		"dim_in_linear",
		"num_components",
		"num_features",
		"nonlinearity",
		"distribution", 0};

	int dim_in_nonlinear;
	int dim_in_linear = 0;
	int num_components = 3;
	int num_features = -1;
	PyObject* nonlinearity = reinterpret_cast<PyObject*>(&LogisticFunction_type);
	PyObject* distribution = reinterpret_cast<PyObject*>(&Bernoulli_type);

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "i|iiiOO", const_cast<char**>(kwlist),
		&dim_in_nonlinear, &dim_in_linear, &num_components, &num_features, &nonlinearity, &distribution))
		return -1;

	if(PyType_Check(nonlinearity)) {
		if(!PyType_IsSubtype(reinterpret_cast<PyTypeObject*>(nonlinearity), &Nonlinearity_type)) {
			PyErr_SetString(PyExc_TypeError, "Nonlinearity should be a subtype of `Nonlinearity`.");
			return -1;
		}

		// create instance of type
		nonlinearity = PyObject_CallObject(nonlinearity, 0);

		if(!nonlinearity)
			return -1;
	} else if(!PyType_IsSubtype(Py_TYPE(nonlinearity), &Nonlinearity_type)) {
		PyErr_SetString(PyExc_TypeError, "Nonlinearity should be of type `Nonlinearity`.");
		return -1;
	}

	if(PyType_Check(distribution)) {
		if(!PyType_IsSubtype(reinterpret_cast<PyTypeObject*>(distribution), &UnivariateDistribution_type)) {
			PyErr_SetString(PyExc_TypeError, "Distribution should be a subtype of `UnivariateDistribution`.");
			return -1;
		}

		// create instance of type
		distribution = PyObject_CallObject(distribution, 0);

		if(!distribution)
			return -1;
	} else if(!PyType_IsSubtype(Py_TYPE(distribution), &UnivariateDistribution_type)) {
		PyErr_SetString(PyExc_TypeError, "Distribution should be of type `UnivariateDistribution`.");
		return -1;
	}

	Py_INCREF(nonlinearity);
	Py_INCREF(distribution);

	// create actual STM instance
	try {
		self->nonlinearity = reinterpret_cast<NonlinearityObject*>(nonlinearity);
		self->distribution = reinterpret_cast<UnivariateDistributionObject*>(distribution);

		self->stm = new STM(
			dim_in_nonlinear,
			dim_in_linear,
			num_components,
			num_features,
			self->nonlinearity->nonlinearity,
			self->distribution->distribution);
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	return 0;
}



PyObject* STM_dim_in_nonlinear(STMObject* self, void*) {
	return PyInt_FromLong(self->stm->dimInNonlinear());
}



PyObject* STM_dim_in_linear(STMObject* self, void*) {
	return PyInt_FromLong(self->stm->dimInLinear());
}



PyObject* STM_num_components(STMObject* self, void*) {
	return PyInt_FromLong(self->stm->numComponents());
}



PyObject* STM_num_features(STMObject* self, void*) {
	return PyInt_FromLong(self->stm->numFeatures());
}



PyObject* STM_sharpness(STMObject* self, void*) {
	return PyFloat_FromDouble(self->stm->sharpness());
}



int STM_set_sharpness(STMObject* self, PyObject* value, void*) {
	try {
		self->stm->setSharpness(PyFloat_AsDouble(value));
	} catch(Exception exception) {
		Py_DECREF(value);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}
	return 0;
}



PyObject* STM_biases(STMObject* self, void*) {
	PyObject* array = PyArray_FromMatrixXd(self->stm->biases());

	// make array immutable
	reinterpret_cast<PyArrayObject*>(array)->flags &= ~NPY_WRITEABLE;

	return array;
}



int STM_set_biases(STMObject* self, PyObject* value, void*) {
	value = PyArray_FROM_OTF(value, NPY_DOUBLE, NPY_IN_ARRAY);

	if(!value) {
		PyErr_SetString(PyExc_TypeError, "Biases should be of type `ndarray`.");
		return -1;
	}

	try {
		self->stm->setBiases(PyArray_ToMatrixXd(value));
	} catch(Exception exception) {
		Py_DECREF(value);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	Py_DECREF(value);

	return 0;
}



PyObject* STM_weights(STMObject* self, void*) {
	PyObject* array = PyArray_FromMatrixXd(self->stm->weights());

	// make array immutable
	reinterpret_cast<PyArrayObject*>(array)->flags &= ~NPY_WRITEABLE;

	return array;
}



int STM_set_weights(STMObject* self, PyObject* value, void*) {
	value = PyArray_FROM_OTF(value, NPY_DOUBLE, NPY_IN_ARRAY);

	if(!value) {
		PyErr_SetString(PyExc_TypeError, "Weights should be of type `ndarray`.");
		return -1;
	}

	try {
		self->stm->setWeights(PyArray_ToMatrixXd(value));
	} catch(Exception exception) {
		Py_DECREF(value);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	Py_DECREF(value);

	return 0;
}



PyObject* STM_features(STMObject* self, void*) {
	PyObject* array = PyArray_FromMatrixXd(self->stm->features());

	// make array immutable
	reinterpret_cast<PyArrayObject*>(array)->flags &= ~NPY_WRITEABLE;

	return array;
}



int STM_set_features(STMObject* self, PyObject* value, void*) {
	value = PyArray_FROM_OTF(value, NPY_DOUBLE, NPY_IN_ARRAY);

	if(!value) {
		PyErr_SetString(PyExc_TypeError, "Features should be of type `ndarray`.");
		return -1;
	}

	try {
		self->stm->setFeatures(PyArray_ToMatrixXd(value));
	} catch(Exception exception) {
		Py_DECREF(value);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	Py_DECREF(value);

	return 0;
}



PyObject* STM_predictors(STMObject* self, void*) {
	PyObject* array = PyArray_FromMatrixXd(self->stm->predictors());

	// make array immutable
	reinterpret_cast<PyArrayObject*>(array)->flags &= ~NPY_WRITEABLE;

	return array;
}



int STM_set_predictors(STMObject* self, PyObject* value, void*) {
	value = PyArray_FROM_OTF(value, NPY_DOUBLE, NPY_IN_ARRAY);

	if(!value) {
		PyErr_SetString(PyExc_TypeError, "Predictors should be of type `ndarray`.");
		return -1;
	}

	try {
		self->stm->setPredictors(PyArray_ToMatrixXd(value));
	} catch(Exception exception) {
		Py_DECREF(value);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	Py_DECREF(value);

	return 0;
}



PyObject* STM_linear_predictor(STMObject* self, void*) {
	PyObject* array = PyArray_FromMatrixXd(self->stm->linearPredictor());

	// make array immutable
	reinterpret_cast<PyArrayObject*>(array)->flags &= ~NPY_WRITEABLE;

	return array;
}



int STM_set_linear_predictor(STMObject* self, PyObject* value, void*) {
	value = PyArray_FROM_OTF(value, NPY_DOUBLE, NPY_IN_ARRAY);

	if(!value) {
		PyErr_SetString(PyExc_TypeError, "Linear predictor should be of type `ndarray`.");
		return -1;
	}

	try {
		self->stm->setLinearPredictor(PyArray_ToMatrixXd(value));
	} catch(Exception exception) {
		Py_DECREF(value);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	Py_DECREF(value);

	return 0;
}



PyObject* STM_nonlinearity(STMObject* self, void*) {
	Py_INCREF(self->nonlinearity);
	return reinterpret_cast<PyObject*>(self->nonlinearity);
}



int STM_set_nonlinearity(STMObject* self, PyObject* nonlinearity, void*) {
	// read arguments
	if(PyType_Check(nonlinearity)) {
		if(!PyType_IsSubtype(reinterpret_cast<PyTypeObject*>(nonlinearity), &Nonlinearity_type)) {
			PyErr_SetString(PyExc_TypeError, "Nonlinearity should be a subtype of `Nonlinearity`.");
			return -1;
		}

		// create instance of type
		nonlinearity = PyObject_CallObject(nonlinearity, 0);
	} else if(!PyType_IsSubtype(Py_TYPE(nonlinearity), &Nonlinearity_type)) {
		PyErr_SetString(PyExc_TypeError, "Nonlinearity should be of type `Nonlinearity`.");
		return -1;
	}

	try {
		Py_INCREF(nonlinearity);
		Py_DECREF(self->nonlinearity);
		self->nonlinearity = reinterpret_cast<NonlinearityObject*>(nonlinearity);
		self->stm->setNonlinearity(reinterpret_cast<NonlinearityObject*>(nonlinearity)->nonlinearity);
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	return 0;
}



PyObject* STM_distribution(STMObject* self, void*) {
	Py_INCREF(self->distribution);
	return reinterpret_cast<PyObject*>(self->distribution);
}



int STM_set_distribution(STMObject* self, PyObject* distribution, void*) {
	// read arguments
	if(PyType_Check(distribution)) {
		if(!PyType_IsSubtype(reinterpret_cast<PyTypeObject*>(distribution), &UnivariateDistribution_type)) {
			PyErr_SetString(PyExc_TypeError, "Distribution should be a subtype of `UnivariateDistribution`.");
			return -1;
		}

		// create instance of type
		distribution = PyObject_CallObject(distribution, 0);
	} else if(!PyType_IsSubtype(Py_TYPE(distribution), &UnivariateDistribution_type)) {
		PyErr_SetString(PyExc_TypeError, "Distribution should be of type `UnivariateDistribution`.");
		return -1;
	}

	try {
		Py_INCREF(distribution);
		Py_DECREF(self->distribution);
		self->distribution = reinterpret_cast<UnivariateDistributionObject*>(distribution);
		self->stm->setDistribution(reinterpret_cast<UnivariateDistributionObject*>(distribution)->distribution);
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	return 0;
}



const char* STM_linear_response_doc =
	"linear_response(self, input)\n"
	"\n"
	"Computes the linear portion of the intenal response, $\\mathbf{v}^\\top \\mathbf{z}$. The input\n"
	"can be the full input or just the linear part, that is, it can be of dimension C{dimIn()} or of\n"
	"dimension C{dimInLinear()}.\n"
	"\n"
	"@type  input: C{ndarray}\n"
	"@param input: inputs stored in columns\n"
	"\n"
	"@rtype: C{ndarray}\n"
	"@return: the output of the linear filter";

PyObject* STM_linear_response(STMObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"input", 0};

	PyObject* input;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O", const_cast<char**>(kwlist), &input))
		return 0;

	// make sure data is stored in NumPy array
	input = PyArray_FROM_OTF(input, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

	if(!input) {
		PyErr_SetString(PyExc_TypeError, "Inputs have to be stored in a NumPy array.");
		return 0;
	}

	try {
		PyObject* result = PyArray_FromMatrixXd(
			self->stm->linearResponse(PyArray_ToMatrixXd(input)));
		Py_DECREF(input);
		return result;
	} catch(Exception exception) {
		Py_DECREF(input);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	return 0;
}



const char* STM_nonlinear_responses_doc =
	"nonlinear_responses(self, input)\n"
	"\n"
	"Computes the nonlinear response of each component,\n"
	"\n"
	"$$\\sum_l \\beta_{kl} (\\mathbf{u}_l^\\top \\mathbf{x})^2 + \\mathbf{w}_k \\mathbf{x} + a_k.$$\n"
	"\n"
	"The input can be the full input or just the nonlinear part, that is, it can be of dimension C{dimIn()} or\n"
	"of dimension C{dimInNonlinear()}.\n"
	"\n"
	"@type  input: C{ndarray}\n"
	"@param input: inputs stored in columns\n"
	"\n"
	"@rtype: C{ndarray}\n"
	"@return: one response for each component and each input";

PyObject* STM_nonlinear_responses(STMObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"input", 0};

	PyObject* input;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O", const_cast<char**>(kwlist), &input))
		return 0;

	// make sure data is stored in NumPy array
	input = PyArray_FROM_OTF(input, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

	if(!input) {
		PyErr_SetString(PyExc_TypeError, "Inputs have to be stored in a NumPy array.");
		return 0;
	}

	try {
		PyObject* result = PyArray_FromMatrixXd(
			self->stm->nonlinearResponses(PyArray_ToMatrixXd(input)));
		Py_DECREF(input);
		return result;
	} catch(Exception exception) {
		Py_DECREF(input);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	return 0;
}



const char* STM_train_doc =
	"train(self, input, output, input_val=None, output_val=None, parameters=None)\n"
	"\n"
	"Fits model parameters to given data using L-BFGS.\n"
	"\n"
	"The following example demonstrates possible parameters and default settings.\n"
	"\n"
	"\t>>> model.train(input, output, parameters={\n"
	"\t>>> \t'verbosity': 0,\n"
	"\t>>> \t'max_iter': 1000,\n"
	"\t>>> \t'threshold': 1e-9,\n"
	"\t>>> \t'num_grad': 20,\n"
	"\t>>> \t'batch_size': 2000,\n"
	"\t>>> \t'callback': None,\n"
	"\t>>> \t'cb_iter': 25,\n"
	"\t>>> \t'val_iter': 5,\n"
	"\t>>> \t'val_look_ahead': 20,\n"
	"\t>>> \t'train_biases': True,\n"
	"\t>>> \t'train_weights': True,\n"
	"\t>>> \t'train_features': True,\n"
	"\t>>> \t'train_predictors': True,\n"
	"\t>>> \t'train_linear_predictor': True,\n"
	"\t>>> \t'regularize_biases': {\n"
	"\t>>> \t\t'strength': 0.,\n"
	"\t>>> \t\t'norm': 'L2'},\n"
	"\t>>> \t'regularize_features': {\n"
	"\t>>> \t\t'strength': 0.,\n"
	"\t>>> \t\t'transform': None,\n"
	"\t>>> \t\t'norm': 'L2'},\n"
	"\t>>> \t'regularize_weights': {\n"
	"\t>>> \t\t'strength': 0.,\n"
	"\t>>> \t\t'norm': 'L2'},\n"
	"\t>>> \t'regularize_predictors': {\n"
	"\t>>> \t\t'strength': 0.,\n"
	"\t>>> \t\t'transform': None,\n"
	"\t>>> \t\t'norm': 'L2'},\n"
	"\t>>> \t'regularize_linear_predictor': {\n"
	"\t>>> \t\t'strength': 0.,\n"
	"\t>>> \t\t'transform': None,\n"
	"\t>>> \t\t'norm': 'L2'},\n"
	"\t>>> })\n"
	"\n"
	"The parameters C{train_biases}, C{train_weights}, and so on can be used to control which\n"
	"parameters will be optimized. Optimization stops after C{max_iter} iterations or if\n"
	"the difference in (penalized) log-likelihood is sufficiently small enough, as specified by\n"
	"C{threshold}. C{num_grad} is the number of gradients used by L-BFGS to approximate the inverse\n"
	"Hessian matrix.\n"
	"\n"
	"Regularization of parameters $\\mathbf{z}$ adds a penalty term\n"
	"\n"
	"$$\\eta ||\\mathbf{A} \\mathbf{z}||_p$$\n"
	"\n"
	"to the average log-likelihood, where $\\eta$ is given by C{strength}, $\\mathbf{A}$ is\n"
	"given by C{transform}, and $p$ is controlled by C{norm}, which has to be either C{'L1'} or C{'L2'}.\n"
	"\n"
	"The parameter C{batch_size} has no effect on the solution of the optimization but\n"
	"can affect speed by reducing the number of cache misses.\n"
	"\n"
	"If a callback function is given, it will be called every C{cb_iter} iterations. The first\n"
	"argument to callback will be the current iteration, the second argument will be a I{copy} of\n"
	"the model.\n"
	"\n"
	"\t>>> def callback(i, stm):\n"
	"\t>>> \tprint i\n"
	"\n"
	"@type  input: C{ndarray}\n"
	"@param input: inputs stored in columns\n"
	"\n"
	"@type  output: C{ndarray}\n"
	"@param output: outputs stored in columns\n"
	"\n"
	"@type  input_val: C{ndarray}\n"
	"@param input_val: inputs used for early stopping based on validation error\n"
	"\n"
	"@type  output_val: C{ndarray}\n"
	"@param output_val: outputs used for early stopping based on validation error\n"
	"\n"
	"@type  parameters: C{dict}\n"
	"@param parameters: a dictionary containing hyperparameters\n"
	"\n"
	"@rtype: C{bool}\n"
	"@return: C{True} if training converged, otherwise C{False}";

PyObject* STM_train(STMObject* self, PyObject* args, PyObject* kwds) {
	return Trainable_train(
		reinterpret_cast<TrainableObject*>(self), 
		args,
		kwds,
		&PyObject_ToSTMParameters);
}



PyObject* STM_parameters(STMObject* self, PyObject* args, PyObject* kwds) {
	return Trainable_parameters(
		reinterpret_cast<TrainableObject*>(self),
		args,
		kwds,
		&PyObject_ToSTMParameters);
}



PyObject* STM_set_parameters(STMObject* self, PyObject* args, PyObject* kwds) {
	return Trainable_set_parameters(
		reinterpret_cast<TrainableObject*>(self),
		args,
		kwds,
		&PyObject_ToSTMParameters);
}



PyObject* STM_parameter_gradient(STMObject* self, PyObject* args, PyObject* kwds) {
	return Trainable_parameter_gradient(
		reinterpret_cast<TrainableObject*>(self),
		args,
		kwds,
		&PyObject_ToSTMParameters);
}



PyObject* STM_fisher_information(STMObject* self, PyObject* args, PyObject* kwds) {
	return Trainable_fisher_information(
		reinterpret_cast<TrainableObject*>(self), 
		args, 
		kwds,
		&PyObject_ToSTMParameters);
}



PyObject* STM_check_gradient(STMObject* self, PyObject* args, PyObject* kwds) {
	return Trainable_check_gradient(
		reinterpret_cast<TrainableObject*>(self),
		args,
		kwds,
		&PyObject_ToSTMParameters);
}



PyObject* STM_check_performance(STMObject* self, PyObject* args, PyObject* kwds) {
	return Trainable_check_performance(
		reinterpret_cast<TrainableObject*>(self),
		args,
		kwds,
		&PyObject_ToSTMParameters);
}



const char* STM_reduce_doc =
	"__reduce__(self)\n"
	"\n"
	"Method used by Pickle.";

PyObject* STM_reduce(STMObject* self, PyObject*) {
	// constructor arguments
	PyObject* args = Py_BuildValue("(iiiiOO)",
		self->stm->dimInNonlinear(),
		self->stm->dimInLinear(),
		self->stm->numComponents(),
		self->stm->numFeatures(),
		self->nonlinearity,
		self->distribution);

	// parameters
	PyObject* biases = STM_biases(self, 0);
	PyObject* weights = STM_weights(self, 0);
	PyObject* features = STM_features(self, 0);
	PyObject* predictors = STM_predictors(self, 0);
	PyObject* linear_predictor = STM_linear_predictor(self, 0);
	PyObject* sharpness = STM_sharpness(self, 0);

	PyObject* state = Py_BuildValue("(OOOOOO)",
		biases, weights, features, predictors, linear_predictor, sharpness);

	Py_DECREF(biases);
	Py_DECREF(weights);
	Py_DECREF(features);
	Py_DECREF(predictors);
	Py_DECREF(linear_predictor);
	Py_DECREF(sharpness);

	PyObject* result = Py_BuildValue("(OOO)", Py_TYPE(self), args, state);

	Py_DECREF(args);
	Py_DECREF(state);

	return result;
}



const char* STM_setstate_doc =
	"__setstate__(self)\n"
	"\n"
	"Method used by Pickle.";

PyObject* STM_setstate(STMObject* self, PyObject* state) {
	PyObject* biases;
	PyObject* weights;
	PyObject* features;
	PyObject* predictors;
	PyObject* linear_predictor;
	PyObject* sharpness = 0;

	if(!PyArg_ParseTuple(state, "(OOOOOO)", &biases, &weights, &features, &predictors, &linear_predictor, &sharpness)) {
		PyErr_Clear();

		// try without sharpness for backwards-compatibility
		if(!PyArg_ParseTuple(state, "(OOOOO)", &biases, &weights, &features, &predictors, &linear_predictor))
			return 0;
	}

	try {
		STM_set_biases(self, biases, 0);
		STM_set_weights(self, weights, 0);
		STM_set_features(self, features, 0);
		STM_set_predictors(self, predictors, 0);
		STM_set_linear_predictor(self, linear_predictor, 0);

		if(sharpness)
			STM_set_sharpness(self, sharpness, 0);
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	Py_INCREF(Py_None);
	return Py_None;
}
