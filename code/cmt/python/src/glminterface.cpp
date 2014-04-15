#include "glminterface.h"
#include "distributioninterface.h"
#include "trainableinterface.h"
#include "callbackinterface.h"

#include "cmt/utils"
using CMT::Exception;

Trainable::Parameters* PyObject_ToGLMParameters(PyObject* parameters) {
	GLM::Parameters* params = dynamic_cast<GLM::Parameters*>(
		PyObject_ToParameters(parameters, new GLM::Parameters));

	// read parameters from dictionary
	if(parameters && parameters != Py_None) {
		PyObject* callback = PyDict_GetItemString(parameters, "callback");
		if(callback)
			if(PyCallable_Check(callback))
				params->callback = new CallbackInterface(&GLM_type, callback);
			else if(callback != Py_None)
				throw Exception("callback should be a function or callable object.");

		PyObject* train_weights = PyDict_GetItemString(parameters, "train_weights");
		if(train_weights)
			if(PyBool_Check(train_weights))
				params->trainWeights = (train_weights == Py_True);
			else
				throw Exception("train_weights should be of type `bool`.");

		PyObject* train_bias = PyDict_GetItemString(parameters, "train_bias");
		if(train_bias)
			if(PyBool_Check(train_bias))
				params->trainBias = (train_bias == Py_True);
			else
				throw Exception("train_bias should be of type `bool`.");

		PyObject* train_nonlinearity = PyDict_GetItemString(parameters, "train_nonlinearity");
		if(train_nonlinearity)
			if(PyBool_Check(train_nonlinearity))
				params->trainNonlinearity = (train_nonlinearity == Py_True);
			else
				throw Exception("train_nonlinearity should be of type `bool`.");

		PyObject* regularize_weights = PyDict_GetItemString(parameters, "regularize_weights");
		if(regularize_weights)
			params->regularizeWeights = PyObject_ToRegularizer(regularize_weights);

		PyObject* regularize_bias = PyDict_GetItemString(parameters, "regularize_bias");
		if(regularize_bias)
			params->regularizeBias = PyObject_ToRegularizer(regularize_bias);

		if(PyDict_GetItemString(parameters, "regularizer"))
			throw Exception("Please use the new interface for specifying regularizer norms.");
	}

	return params;
}



const char* GLM_doc =
	"An implementation of generalized linear models.\n"
	"\n"
	"$$p(y \\mid \\mathbf{x}) = q(y \\mid g(\\mathbf{w}^\\top \\mathbf{x} + b)),$$\n"
	"\n"
	"where $q$ is typically from the exponential family and $g$ is some nonlinearity\n"
	"(inverse link function) which has to be specified.\n"
	"\n"
	"To perform logistic regression, for example, define a GLM with L{LogisticFunction<nonlinear.LogisticFunction>}\n"
	"and L{Bernoulli<models.Bernoulli>} distribution,\n"
	"\n"
	"\t>>> glm = GLM(inputs.shape[0], LogisticFunction, Bernoulli)\n"
	"\t>>> glm.train(inputs, outputs)\n"
	"\n"
	"To access the linear filter $\\mathbf{w}$, the bias $b$, the nonlinearity $g$ or the\n"
	"output distribution $f$, use\n"
	"\n"
	"\t>>> glm.weights\n"
	"\t>>> glm.bias\n"
	"\t>>> glm.nonlinearity\n"
	"\t>>> glm.distribution\n"
	"\n"
	"@type  dim_in: C{int}\n"
	"@param dim_in: dimensionality of input\n"
	"\n"
	"@type  nonlinearity: L{Nonlinearity<nonlinear.Nonlinearity>}/C{type}\n"
	"@param nonlinearity: nonlinearity applied to output of linear filter, $g$ (default: L{LogisticFunction<nonlinear.LogisticFunction>})\n"
	"\n"
	"@type  distribution: L{UnivariateDistribution<models.UnivariateDistribution>}/C{type}\n"
	"@param distribution: distribution of outputs, $q$ (default: L{Bernoulli<models.Bernoulli>})";

int GLM_init(GLMObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"dim_in", "nonlinearity", "distribution", 0};

	int dim_in;
	PyObject* nonlinearity = reinterpret_cast<PyObject*>(&LogisticFunction_type);
	PyObject* distribution = reinterpret_cast<PyObject*>(&Bernoulli_type);

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "i|OO", const_cast<char**>(kwlist),
		&dim_in, &nonlinearity, &distribution))
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

	// create actual GLM instance
	try {
		self->nonlinearity = reinterpret_cast<NonlinearityObject*>(nonlinearity);
		self->distribution = reinterpret_cast<UnivariateDistributionObject*>(distribution);

		self->glm = new GLM(
			dim_in,
			self->nonlinearity->nonlinearity,
			self->distribution->distribution);
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	return 0;
}



void GLM_dealloc(GLMObject* self) {
	// delete actual instance
	if(self->glm && self->owner) {
		delete self->glm;

		Py_DECREF(self->nonlinearity);
		Py_DECREF(self->distribution);
	}

	// delete GLMObject
	self->ob_type->tp_free(reinterpret_cast<PyObject*>(self));
}



PyObject* GLM_weights(GLMObject* self, void*) {
	PyObject* array = PyArray_FromMatrixXd(self->glm->weights());

	// make array immutable
	reinterpret_cast<PyArrayObject*>(array)->flags &= ~NPY_WRITEABLE;

	return array;
}



int GLM_set_weights(GLMObject* self, PyObject* value, void*) {
	value = PyArray_FROM_OTF(value, NPY_DOUBLE, NPY_IN_ARRAY);

	if(!value) {
		PyErr_SetString(PyExc_TypeError, "Weights should be of type `ndarray`.");
		return -1;
	}

	try {
		self->glm->setWeights(PyArray_ToMatrixXd(value));
	} catch(Exception exception) {
		Py_DECREF(value);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	Py_DECREF(value);

	return 0;
}



PyObject* GLM_bias(GLMObject* self, void*) {
	return PyFloat_FromDouble(self->glm->bias());
}



int GLM_set_bias(GLMObject* self, PyObject* value, void*) {
	double bias = PyFloat_AsDouble(value);
	
	if(PyErr_Occurred()) {
		PyErr_SetString(PyExc_TypeError, "Bias should be a `float`.");
		return -1;
	}

	try {
		self->glm->setBias(bias);
	} catch(Exception exception) {
		Py_DECREF(value);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	Py_DECREF(value);

	return 0;
}



PyObject* GLM_nonlinearity(GLMObject* self, void*) {
	Py_INCREF(self->nonlinearity);
	return reinterpret_cast<PyObject*>(self->nonlinearity);
}



int GLM_set_nonlinearity(GLMObject* self, PyObject* nonlinearity, void*) {
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
		self->glm->setNonlinearity(reinterpret_cast<NonlinearityObject*>(nonlinearity)->nonlinearity);
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	return 0;
}



PyObject* GLM_distribution(GLMObject* self, void*) {
	Py_INCREF(self->distribution);
	return reinterpret_cast<PyObject*>(self->distribution);
}



int GLM_set_distribution(GLMObject* self, PyObject* distribution, void*) {
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
		self->glm->setDistribution(reinterpret_cast<UnivariateDistributionObject*>(distribution)->distribution);
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	return 0;
}



const char* GLM_train_doc =
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
	"\t>>> \t'train_weights': True,\n"
	"\t>>> \t'train_bias': True,\n"
	"\t>>> \t'train_nonlinearity': False,\n"
	"\t>>> \t'regularize_weights': {\n"
	"\t>>> \t\t'strength': 0.,\n"
	"\t>>> \t\t'transform': None,\n"
	"\t>>> \t\t'norm': 'L2'},\n"
	"\t>>> \t'regularize_bias': {\n"
	"\t>>> \t\t'strength': 0.,\n"
	"\t>>> \t\t'norm': 'L2'},\n"
	"\t>>> })\n"
	"\n"
	"The optimization stops after C{max_iter} iterations or if the difference in\n"
	"(penalized) log-likelihood is sufficiently small enough, as specified by\n"
	"C{threshold}. C{num_grad} is the number of gradients used by L-BFGS to approximate\n"
	"the inverse Hessian matrix.\n"
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
	"\t>>> def callback(i, glm):\n"
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

PyObject* GLM_train(GLMObject* self, PyObject* args, PyObject* kwds) {
	return Trainable_train(
		reinterpret_cast<TrainableObject*>(self), 
		args, 
		kwds,
		&PyObject_ToGLMParameters);
}



PyObject* GLM_parameters(GLMObject* self, PyObject* args, PyObject* kwds) {
	return Trainable_parameters(
		reinterpret_cast<TrainableObject*>(self),
		args,
		kwds,
		&PyObject_ToGLMParameters);
}



PyObject* GLM_set_parameters(GLMObject* self, PyObject* args, PyObject* kwds) {
	return Trainable_set_parameters(
		reinterpret_cast<TrainableObject*>(self),
		args,
		kwds,
		&PyObject_ToGLMParameters);
}



PyObject* GLM_parameter_gradient(GLMObject* self, PyObject* args, PyObject* kwds) {
	return Trainable_parameter_gradient(
		reinterpret_cast<TrainableObject*>(self), 
		args, 
		kwds,
		&PyObject_ToGLMParameters);
}



PyObject* GLM_fisher_information(GLMObject* self, PyObject* args, PyObject* kwds) {
	return Trainable_fisher_information(
		reinterpret_cast<TrainableObject*>(self), 
		args, 
		kwds,
		&PyObject_ToGLMParameters);
}



PyObject* GLM_check_gradient(GLMObject* self, PyObject* args, PyObject* kwds) {
	return Trainable_check_gradient(
		reinterpret_cast<TrainableObject*>(self), 
		args, 
		kwds,
		&PyObject_ToGLMParameters);
}



PyObject* GLM_check_performance(GLMObject* self, PyObject* args, PyObject* kwds) {
	return Trainable_check_performance(
		reinterpret_cast<TrainableObject*>(self),
		args,
		kwds,
		&PyObject_ToGLMParameters);
}



const char* GLM_reduce_doc =
	"__reduce__(self)\n"
	"\n"
	"Method used by Pickle.";

PyObject* GLM_reduce(GLMObject* self, PyObject*) {
	// constructor arguments
	PyObject* args = Py_BuildValue("(iOO)",
		self->glm->dimIn(),
		self->nonlinearity,
		self->distribution);
	PyObject* weights = GLM_weights(self, 0);
	PyObject* state = Py_BuildValue("(Od)", weights, self->glm->bias());
	PyObject* result = Py_BuildValue("(OOO)", Py_TYPE(self), args, state);

	Py_DECREF(weights);
	Py_DECREF(args);
	Py_DECREF(state);

	return result;
}



const char* GLM_setstate_doc =
	"__setstate__(self)\n"
	"\n"
	"Method used by Pickle.";

PyObject* GLM_setstate(GLMObject* self, PyObject* state) {
	PyObject* weights;
	double bias;

	if(!PyArg_ParseTuple(state, "(Od)", &weights, &bias))
		return 0;

	try {
		GLM_set_weights(self, weights, 0);
		self->glm->setBias(bias);
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	Py_INCREF(Py_None);
	return Py_None;
}
