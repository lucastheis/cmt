#include "distributioninterface.h"
#include "trainableinterface.h"
#include "callbackinterface.h"
#include "glminterface.h"

#include "exception.h"
using CMT::Exception;

PyObject* Nonlinearity_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
	PyObject* self = type->tp_alloc(type, 0);

	if(self) {
		reinterpret_cast<NonlinearityObject*>(self)->nonlinearity = 0;
		reinterpret_cast<NonlinearityObject*>(self)->owner = true;
	}

	return self;
}



const char* Nonlinearity_doc =
	"Abstract base class for nonlinear functions usable by L{GLM}.";

int Nonlinearity_init(
	NonlinearityObject* self,
	PyObject* args,
	PyObject* kwds)
{
	PyErr_SetString(PyExc_NotImplementedError, "This is an abstract class.");
	return -1;
}



void Nonlinearity_dealloc(NonlinearityObject* self) {
	// delete actual instance
	if(self->owner)
		delete self->nonlinearity;

	// delete NonlinearityObject
	self->ob_type->tp_free(reinterpret_cast<PyObject*>(self));
}



PyObject* Nonlinearity_call(NonlinearityObject* self, PyObject* args, PyObject*) {
	PyObject* x = 0;

	if(!PyArg_ParseTuple(args, "O", &x))
		return 0;

	x = PyArray_FROM_OTF(x, NPY_DOUBLE, NPY_IN_ARRAY);

	if(!x) {
		PyErr_SetString(PyExc_TypeError, "Data should be of type `ndarray`.");
		return 0;
	}

	try {
		MatrixXd output = (*self->nonlinearity)(PyArray_ToMatrixXd(x));
		Py_DECREF(x);
		return PyArray_FromMatrixXd(output);
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		Py_DECREF(x);
		return 0;
	}

	Py_DECREF(x);
	return 0;
}



const char* LogisticFunction_doc =
	"The sigmoidal logistic function.\n"
	"\n"
	"$$f(x) = (1 + e^{-x})^{-1}$$";

int LogisticFunction_init(LogisticFunctionObject* self, PyObject*, PyObject*) {
	try {
		self->nonlinearity = new LogisticFunction;
		return 0;
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	return -1;
}



const char* LogisticFunction_reduce_doc =
	"__reduce__(self)\n"
	"\n"
	"Method used by Pickle.";

PyObject* LogisticFunction_reduce(LogisticFunctionObject* self, PyObject*) {
	PyObject* args = Py_BuildValue("()");
	PyObject* result = Py_BuildValue("(OO)", Py_TYPE(self), args);

	Py_DECREF(args);

	return result;
}



const char* UnivariateDistribution_doc =
	"Abstract base class for univariate distributions usable by L{GLM}.";

int UnivariateDistribution_init(
	UnivariateDistributionObject* self,
	PyObject* args,
	PyObject* kwds)
{
	PyErr_SetString(PyExc_NotImplementedError, "This is an abstract class.");
	return -1;
}



const char* Bernoulli_doc =
	"The Bernoulli distribution.\n"
	"\n"
	"$$p(y) = \\rho^y (1 - \\rho)^{1 - y}$$\n"
	"\n"
	"@type  probability: float\n"
	"@param probability: probability of generating a 1, $\\rho$";

int Bernoulli_init(BernoulliObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"probability", 0};

	double prob = 0.5;

	if(!PyArg_ParseTupleAndKeywords(args, kwds, "|d", const_cast<char**>(kwlist), &prob))
		return -1;

	try {
		self->distribution = new Bernoulli(prob);
		return 0;
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	return -1;
}



const char* Bernoulli_reduce_doc =
	"__reduce__(self)\n"
	"\n"
	"Method used by Pickle.";

PyObject* Bernoulli_reduce(BernoulliObject* self, PyObject*) {
	PyObject* args = Py_BuildValue("(d)", self->distribution->probability());
	PyObject* result = Py_BuildValue("(OO)", Py_TYPE(self), args);

	Py_DECREF(args);

	return result;
}



const char* GLM_doc =
	"An implementation of generalized linear models.\n"
	"\n"
	"$$p(y \\mid \\mathbf{x}) = q(y \\mid g(\\mathbf{x}^\\top \\mathbf{x})),$$\n"
	"\n"
	"where $q$ is typically from the exponential family and $g$ is some nonlinearity\n"
	"(inverse link function) which has to be specified.\n"
	"\n"
	"To perform logistic regression, for example, define a GLM with L{LogisticFunction}\n"
	"and L{Bernoulli} distribution,\n"
	"\n"
	"\t>>> glm = GLM(inputs.shape[0], LogisticFunction, Bernoulli)\n"
	"\t>>> glm.train(inputs, outputs)\n"
	"\n"
	"To access the linear filter, $\\mathbf{w}$, use\n"
	"\n"
	"\t>>> glm.weights\n"
	"\n"
	"@type  dim_in: integer\n"
	"@param dim_in: dimensionality of input\n"
	"\n"
	"@type  nonlinearity: L{Nonlinearity}/C{type}\n"
	"@param nonlinearity: nonlinearity applied to output of linear filter, $g$\n"
	"\n"
	"@type  distribution: L{UnivariateDistribution}/C{type}\n"
	"@param distribution: distribution of outputs";

int GLM_init(GLMObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"dim_in", "nonlinearity", "distribution", 0};

	int dim_in;
	PyObject* nonlinearity;
	PyObject* distribution;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "iOO", const_cast<char**>(kwlist),
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

	Py_INCREF(nonlinearity);

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
	if(self->owner)
		delete self->glm;

	Py_DECREF(self->nonlinearity);
	Py_DECREF(self->distribution);

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



Trainable::Parameters* PyObject_ToGLMParameters(PyObject* parameters) {
	Trainable::Parameters* params = PyObject_ToParameters(parameters);

	if(parameters && parameters != Py_None) {
		PyObject* callback = PyDict_GetItemString(parameters, "callback");
		if(callback)
			if(PyCallable_Check(callback))
				params->callback = new CallbackInterface(&GLM_type, callback);
			else if(callback != Py_None)
				throw Exception("callback should be a function or callable object.");
	}

	return params;
}



PyObject* GLM_parameter_gradient(GLMObject* self, PyObject* args, PyObject* kwds) {
	return Trainable_parameter_gradient(
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



PyObject* GLM_train(GLMObject* self, PyObject* args, PyObject* kwds) {
	return Trainable_train(
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
	PyObject* state = Py_BuildValue("(O)", weights);
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

	if(!PyArg_ParseTuple(state, "(O)", &weights))
		return 0;

	try {
		GLM_set_weights(self, weights, 0);
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	Py_INCREF(Py_None);
	return Py_None;
}
