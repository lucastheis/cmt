#include "exception.h"
#include "distributioninterface.h"
#include "glminterface.h"

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
		if(PyType_IsSubtype(reinterpret_cast<PyTypeObject*>(nonlinearity), &Nonlinearity_type)) {
			PyErr_SetString(PyExc_TypeError, "Nonlinearity should be a subtype of `Nonlinearity`.");
			return 0;
		}

		// create instance of type
		nonlinearity = Nonlinearity_new(reinterpret_cast<PyTypeObject*>(nonlinearity), 0, 0);
	} else if(!PyType_IsSubtype(Py_TYPE(nonlinearity), &Nonlinearity_type)) {
		PyErr_SetString(PyExc_TypeError, "Nonlinearity should be of type `Nonlinearity`.");
		return 0;
	}

	if(PyType_Check(distribution)) {
		if(PyType_IsSubtype(reinterpret_cast<PyTypeObject*>(distribution), &UnivariateDistribution_type)) {
			PyErr_SetString(PyExc_TypeError, "Distribution should be a subtype of `UnivariateDistribution`.");
			return 0;
		}

		// create instance of type
		distribution = Distribution_new(reinterpret_cast<PyTypeObject*>(distribution), 0, 0);
	} else if(!PyType_IsSubtype(Py_TYPE(distribution), &UnivariateDistribution_type)) {
		PyErr_SetString(PyExc_TypeError, "Distribution should be of type `UnivariateDistribution`.");
		return 0;
	}

	// create actual MCBM instance
	try {
		Py_INCREF(nonlinearity);
		Py_INCREF(distribution);

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
