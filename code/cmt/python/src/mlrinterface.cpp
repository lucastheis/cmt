#include "mlrinterface.h"
#include "distributioninterface.h"
#include "trainableinterface.h"
#include "callbackinterface.h"

#include "cmt/utils"
using CMT::Exception;

Trainable::Parameters* PyObject_ToMLRParameters(PyObject* parameters) {
	MLR::Parameters* params = dynamic_cast<MLR::Parameters*>(
		PyObject_ToParameters(parameters, new MLR::Parameters));

	// read parameters from dictionary
	if(parameters && parameters != Py_None) {
		PyObject* callback = PyDict_GetItemString(parameters, "callback");
		if(callback)
			if(PyCallable_Check(callback))
				params->callback = new CallbackInterface(&MLR_type, callback);
			else if(callback != Py_None)
				throw Exception("callback should be a function or callable object.");

		PyObject* train_weights = PyDict_GetItemString(parameters, "train_weights");
		if(train_weights)
			if(PyBool_Check(train_weights))
				params->trainWeights = (train_weights == Py_True);
			else
				throw Exception("train_weights should be of type `bool`.");

		PyObject* train_biases = PyDict_GetItemString(parameters, "train_biases");
		if(train_biases)
			if(PyBool_Check(train_biases))
				params->trainBiases = (train_biases == Py_True);
			else
				throw Exception("train_biases should be of type `bool`.");

		PyObject* regularize_weights = PyDict_GetItemString(parameters, "regularize_weights");
		if(regularize_weights)
			params->regularizeWeights = PyObject_ToRegularizer(regularize_weights);

		PyObject* regularize_biases = PyDict_GetItemString(parameters, "regularize_biases");
		if(regularize_biases)
			params->regularizeBiases = PyObject_ToRegularizer(regularize_biases);

	}

	return params;
}



const char* MLR_doc =
	"An implementation of multinomial logistic regression (also known as softmax regression).\n"
	"\n"
	"$$p(\\mathbf{y} \\mid \\mathbf{x}) = \\frac{\\exp(\\mathbf{w}_i^\\top \\mathbf{x} + b_i)}{\\sum_j \\exp(\\mathbf{w}_j^\\top \\mathbf{x} + b_j)},$$\n"
	"\n"
	"for $\\mathbf{y} \\in \\{0, 1\\}^N$ with $y_i = 1$ and $y_j = 0$ for all $j \\neq i$.\n"
	"\n"
	"To access linear filters $\\mathbf{w}_i$ and biases $b$, use\n"
	"\n"
	"\t>>> mlr = MLR(M, N)\n"
	"\t>>> mlr.weights\n"
	"\t>>> mlr.biases\n"
	"\n"
	"@type  dim_in: C{int}\n"
	"@param dim_in: dimensionality of input\n"
	"\n"
	"@type  dim_out: C{int}\n"
	"@param dim_out: number of possible outputs (one-hot encoded)";

int MLR_init(MLRObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"dim_in", "dim_out", 0};

	int dim_in;
	int dim_out = 2;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "i|i", const_cast<char**>(kwlist),
		&dim_in, &dim_out))
		return -1;

	// create actual MLR instance
	try {
		self->mlr = new MLR(dim_in, dim_out);
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	return 0;
}



void MLR_dealloc(MLRObject* self) {
	// delete actual instance
	if(self->mlr && self->owner)
		delete self->mlr;

	// delete MLRObject
	self->ob_type->tp_free(reinterpret_cast<PyObject*>(self));
}



PyObject* MLR_weights(MLRObject* self, void*) {
	PyObject* array = PyArray_FromMatrixXd(self->mlr->weights());

	// make array immutable
	reinterpret_cast<PyArrayObject*>(array)->flags &= ~NPY_WRITEABLE;

	return array;
}



int MLR_set_weights(MLRObject* self, PyObject* value, void*) {
	value = PyArray_FROM_OTF(value, NPY_DOUBLE, NPY_IN_ARRAY);

	if(!value) {
		PyErr_SetString(PyExc_TypeError, "Weights should be of type `ndarray`.");
		return -1;
	}

	try {
		self->mlr->setWeights(PyArray_ToMatrixXd(value));
	} catch(Exception exception) {
		Py_DECREF(value);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	Py_DECREF(value);

	return 0;
}



PyObject* MLR_biases(MLRObject* self, void*) {
	PyObject* array = PyArray_FromMatrixXd(self->mlr->biases());

	// make array immutable
	reinterpret_cast<PyArrayObject*>(array)->flags &= ~NPY_WRITEABLE;

	return array;
}



int MLR_set_biases(MLRObject* self, PyObject* value, void*) {
	value = PyArray_FROM_OTF(value, NPY_DOUBLE, NPY_IN_ARRAY);

	if(!value) {
		PyErr_SetString(PyExc_TypeError, "Biases should be of type `ndarray`.");
		return -1;
	}

	try {
		self->mlr->setBiases(PyArray_ToMatrixXd(value));
	} catch(Exception exception) {
		Py_DECREF(value);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	Py_DECREF(value);

	return 0;
}



const char* MLR_train_doc =
	"train(self, input, output, input_val=None, output_val=None, parameters=None)\n"
	"\n"
	"Fits model parameters to given data using L-BFGS. Because the parameters are redundant,\n"
	"the filter and bias belonging to the first output are fixed and only the remaining\n"
	"parameters are optimized.\n"
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
	"\t>>> \t'train_biases': True,\n"
	"\t>>> \t'regularize_weights': {\n"
	"\t>>> \t\t'strength': 0.,\n"
	"\t>>> \t\t'transform': None,\n"
	"\t>>> \t\t'norm': 'L2'},\n"
	"\t>>> \t'regularize_biases': {\n"
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
	"\t>>> def callback(i, mlr):\n"
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

PyObject* MLR_train(MLRObject* self, PyObject* args, PyObject* kwds) {
	return Trainable_train(
		reinterpret_cast<TrainableObject*>(self), 
		args, 
		kwds,
		&PyObject_ToMLRParameters);
}



PyObject* MLR_parameters(MLRObject* self, PyObject* args, PyObject* kwds) {
	return Trainable_parameters(
		reinterpret_cast<TrainableObject*>(self),
		args,
		kwds,
		&PyObject_ToMLRParameters);
}



PyObject* MLR_set_parameters(MLRObject* self, PyObject* args, PyObject* kwds) {
	return Trainable_set_parameters(
		reinterpret_cast<TrainableObject*>(self),
		args,
		kwds,
		&PyObject_ToMLRParameters);
}



PyObject* MLR_parameter_gradient(MLRObject* self, PyObject* args, PyObject* kwds) {
	return Trainable_parameter_gradient(
		reinterpret_cast<TrainableObject*>(self), 
		args, 
		kwds,
		&PyObject_ToMLRParameters);
}



PyObject* MLR_check_gradient(MLRObject* self, PyObject* args, PyObject* kwds) {
	return Trainable_check_gradient(
		reinterpret_cast<TrainableObject*>(self), 
		args, 
		kwds,
		&PyObject_ToMLRParameters);
}



PyObject* MLR_check_performance(MLRObject* self, PyObject* args, PyObject* kwds) {
	return Trainable_check_performance(
		reinterpret_cast<TrainableObject*>(self),
		args,
		kwds,
		&PyObject_ToMLRParameters);
}



const char* MLR_reduce_doc =
	"__reduce__(self)\n"
	"\n"
	"Method used by Pickle.";

PyObject* MLR_reduce(MLRObject* self, PyObject*) {
	// constructor arguments
	PyObject* args = Py_BuildValue("(ii)", self->mlr->dimIn(), self->mlr->dimOut());
	PyObject* weights = MLR_weights(self, 0);
	PyObject* biases = MLR_biases(self, 0);
	PyObject* state = Py_BuildValue("(OO)", weights, biases);
	PyObject* result = Py_BuildValue("(OOO)", Py_TYPE(self), args, state);

	Py_DECREF(weights);
	Py_DECREF(biases);
	Py_DECREF(args);
	Py_DECREF(state);

	return result;
}



const char* MLR_setstate_doc =
	"__setstate__(self)\n"
	"\n"
	"Method used by Pickle.";

PyObject* MLR_setstate(MLRObject* self, PyObject* state) {
	PyObject* weights;
	PyObject* biases;

	if(!PyArg_ParseTuple(state, "(OO)", &weights, &biases))
		return 0;

	try {
		MLR_set_weights(self, weights, 0);
		MLR_set_biases(self, biases, 0);
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	Py_INCREF(Py_None);
	return Py_None;
}
