#include "conditionaldistributioninterface.h"
#include "callbackinterface.h"
#include "trainableinterface.h"
#include "patchmodelinterface.h"
#include "preconditionerinterface.h"
#include "mcbminterface.h"

#include <utility>
using std::make_pair;

#include "Eigen/Core"
using Eigen::Map;

#include "cmt/utils"
using CMT::Exception;

#include "cmt/tools"
using CMT::Tuples;

Trainable::Parameters* PyObject_ToMCBMParameters(PyObject* parameters) {
	MCBM::Parameters* params = dynamic_cast<MCBM::Parameters*>(
		PyObject_ToParameters(parameters, new MCBM::Parameters));

	// read parameters from dictionary
	if(parameters && parameters != Py_None) {
		PyObject* callback = PyDict_GetItemString(parameters, "callback");
		if(callback)
			if(PyCallable_Check(callback))
				params->callback = new CallbackInterface(&MCBM_type, callback);
			else if(callback != Py_None)
				throw Exception("callback should be a function or callable object.");

		PyObject* train_priors = PyDict_GetItemString(parameters, "train_priors");
		if(train_priors)
			if(PyBool_Check(train_priors))
				params->trainPriors = (train_priors == Py_True);
			else
				throw Exception("train_priors should be of type `bool`.");

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

		PyObject* train_input_bias = PyDict_GetItemString(parameters, "train_input_bias");
		if(train_input_bias)
			if(PyBool_Check(train_input_bias))
				params->trainInputBias = (train_input_bias == Py_True);
			else
				throw Exception("train_input_bias should be of type `bool`.");

		PyObject* train_output_bias = PyDict_GetItemString(parameters, "train_output_bias");
		if(train_output_bias)
			if(PyBool_Check(train_output_bias))
				params->trainOutputBias = (train_output_bias == Py_True);
			else
				throw Exception("train_output_bias should be of type `bool`.");

		PyObject* regularize_features = PyDict_GetItemString(parameters, "regularize_features");
		if(regularize_features)
			params->regularizeFeatures = PyObject_ToRegularizer(regularize_features);

		PyObject* regularize_predictors = PyDict_GetItemString(parameters, "regularize_predictors");
		if(regularize_predictors)
			params->regularizePredictors = PyObject_ToRegularizer(regularize_predictors);

		PyObject* regularize_weights = PyDict_GetItemString(parameters, "regularize_weights");
		if(regularize_weights)
			params->regularizeWeights = PyObject_ToRegularizer(regularize_weights);
	}

	return params;
}



const char* MCBM_doc =
	"An implementation of a mixture of conditional Boltzmann machines.\n"
	"\n"
	"The conditional distribution defined by the model is\n"
	"\n"
	"$$p(y \\mid \\mathbf{x}) \\propto \\sum_{c} \\exp\\left(\\eta_c + \\sum_i \\beta_{ci} \\left(\\mathbf{b}_i^\\top \\mathbf{x}\\right)^2 + \\mathbf{w}_c^\\top \\mathbf{x} + \\mathbf{y}_c^\\top \\mathbf{A}_c \\mathbf{x} + v_c y\\right),$$\n"
	"\n"
	"where $y \\in \\{0, 1\\}$ and $\\mathbf{x} \\in \\mathbb{R}^N$ (although typically $\\mathbf{x} \\in \\{0, 1\\}^N$).\n"
	"\n"
	"To create an MCBM with $N$-dimensional inputs and, for example, 8 components and 100 features $\\mathbf{b}_i$, use\n"
	"\n"
	"\t>>> mcbm = MCBM(N, 8, 100)\n"
	"\n"
	"To access the different parameters, you can use\n"
	"\n"
	"\t>>> mcbm.priors\n"
	"\t>>> mcbm.eights\n"
	"\t>>> mcbm.features\n"
	"\t>>> mcbm.predictors\n"
	"\t>>> mcbm.input_bias\n"
	"\t>>> mcbm.output_bias\n"
	"\n"
	"which correspond to $\\eta_{c}$, $\\beta_{ci}$, $\\mathbf{b}_i$, $\\mathbf{A}_c$, $\\mathbf{w}_c$,"
	"and $v_c$, respectively.\n"
	"\n"
	"@type  dim_in: C{int}\n"
	"@param dim_in: dimensionality of input\n"
	"\n"
	"@type  num_components: C{int}\n"
	"@param num_components: number of components\n"
	"\n"
	"@type  num_features: C{int}\n"
	"@param num_features: number of quadratic features";

int MCBM_init(MCBMObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"dim_in", "num_components", "num_features", 0};

	int dim_in;
	int num_components = 8;
	int num_features = -1;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "i|ii", const_cast<char**>(kwlist),
		&dim_in, &num_components, &num_features))
		return -1;

	// create actual MCBM instance
	try {
		self->mcbm = new MCBM(dim_in, num_components, num_features);
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	return 0;
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
	"\t>>> \t'threshold': 1e-9,\n"
	"\t>>> \t'num_grad': 20,\n"
	"\t>>> \t'batch_size': 2000,\n"
	"\t>>> \t'callback': None,\n"
	"\t>>> \t'cb_iter': 25,\n"
	"\t>>> \t'val_iter': 5,\n"
	"\t>>> \t'val_look_ahead': 20,\n"
	"\t>>> \t'train_priors': True,\n"
	"\t>>> \t'train_weights': True,\n"
	"\t>>> \t'train_features': True,\n"
	"\t>>> \t'train_predictors': True,\n"
	"\t>>> \t'train_input_bias': True,\n"
	"\t>>> \t'train_output_bias': True,\n"
	"\t>>> \t'regularize_features': {\n"
	"\t>>> \t\t'strength': 0.,\n"
	"\t>>> \t\t'transform': None,\n"
	"\t>>> \t\t'norm': 'L2'},\n"
	"\t>>> \t'regularize_weights': {\n"
	"\t>>> \t\t'strength': 0.,\n"
	"\t>>> \t\t'transform': None,\n"
	"\t>>> \t\t'norm': 'L2'},\n"
	"\t>>> \t'regularize_predictors': {\n"
	"\t>>> \t\t'strength': 0.,\n"
	"\t>>> \t\t'transform': None,\n"
	"\t>>> \t\t'norm': 'L2'},\n"
	"\t>>> })\n"
	"\n"
	"The parameters C{train_priors}, C{train_weights}, and so on can be used to control which "
	"parameters will be optimized. Optimization stops after C{max_iter} iterations or if "
	"the difference in (penalized) log-likelihood is sufficiently small enough, as specified by "
	"C{threshold}. C{num_grad} is the number of gradients used by L-BFGS to approximate the inverse "
	"Hessian matrix.\n"
	"\n"
	"Regularization of parameters $\\mathbf{z}$ adds a penalty term\n"
	"\n"
	"$$\\eta ||\\mathbf{A} \\mathbf{z}||_p$$\n"
	"\n"
	"to the average log-likelihood, where $\\eta$ is given by C{strength}, $\\mathbf{A}$ is\n"
	"given by C{transform}, and $p$ is controlled by C{norm}, which has to be either C{'L1'} or C{'L2'}.\n"
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

PyObject* MCBM_train(MCBMObject* self, PyObject* args, PyObject* kwds) {
	return Trainable_train(
		reinterpret_cast<TrainableObject*>(self), 
		args,
		kwds,
		&PyObject_ToMCBMParameters);
}



const char* MCBM_sample_posterior_doc =
	"sample_posterior(self, input, output)\n"
	"\n"
	"Samples component labels $c$ from the posterior $p(c \\mid \\mathbf{x}, \\mathbf{y})$.\n"
	"\n"
	"@type  input: C{ndarray}\n"
	"@param input: inputs stored in columns\n"
	"\n"
	"@type  output: C{ndarray}\n"
	"@param output: inputs stored in columns\n"
	"\n"
	"@rtype: C{ndarray}\n"
	"@return: an integer array containing a sampled index for each input and output pair";

PyObject* MCBM_sample_posterior(MCBMObject* self, PyObject* args, PyObject* kwds) {
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
		Py_XDECREF(input);
		Py_XDECREF(output);
		PyErr_SetString(PyExc_TypeError, "Data has to be stored in NumPy arrays.");
		return 0;
	}

	try {
		PyObject* result = PyArray_FromMatrixXi(
			self->mcbm->samplePosterior(
				PyArray_ToMatrixXd(input),
				PyArray_ToMatrixXd(output)));
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



PyObject* MCBM_parameters(MCBMObject* self, PyObject* args, PyObject* kwds) {
	return Trainable_parameters(
		reinterpret_cast<TrainableObject*>(self), 
		args, 
		kwds,
		&PyObject_ToMCBMParameters);
}



PyObject* MCBM_set_parameters(MCBMObject* self, PyObject* args, PyObject* kwds) {
	return Trainable_set_parameters(
		reinterpret_cast<TrainableObject*>(self), 
		args, 
		kwds,
		&PyObject_ToMCBMParameters);
}



PyObject* MCBM_parameter_gradient(MCBMObject* self, PyObject* args, PyObject* kwds) {
	return Trainable_parameter_gradient(
		reinterpret_cast<TrainableObject*>(self), 
		args, 
		kwds,
		&PyObject_ToMCBMParameters);
}



PyObject* MCBM_check_gradient(MCBMObject* self, PyObject* args, PyObject* kwds) {
	return Trainable_check_gradient(
		reinterpret_cast<TrainableObject*>(self), 
		args, 
		kwds,
		&PyObject_ToMCBMParameters);
}



PyObject* MCBM_check_performance(MCBMObject* self, PyObject* args, PyObject* kwds) {
	return Trainable_check_performance(
		reinterpret_cast<TrainableObject*>(self), 
		args, 
		kwds,
		&PyObject_ToMCBMParameters);
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
	"@type  rows: C{int}\n"
	"@param rows: number of rows of the image patch\n"
	"\n"
	"@type  cols: C{int}\n"
	"@param cols: number of columns of the image patch\n"
	"\n"
	"@type  input_mask: C{ndarray}\n"
	"@param input_mask: a Boolean array describing the input pixels\n"
	"\n"
	"@type  output_mask: C{ndarray}\n"
	"@param output_mask: a Boolean array describing the output pixels\n"
	"\n"
	"@type  order: C{list}\n"
	"@param order: list of tuples indicating order of pixels\n"
	"\n"
	"@type  model: L{MCBM}\n"
	"@param model: model used as a template to initialize all conditional distributions\n"
	"\n"
	"@type  max_pcs: C{int}\n"
	"@param max_pcs: can be used to reduce dimensionality of inputs to conditional models";

int PatchMCBM_init(PatchMCBMObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"rows", "cols", "input_mask", "output_mask", "order", "model", "max_pcs", 0};

	int rows;
	int cols;
	PyObject* input_mask = 0;
	PyObject* output_mask = 0;
	PyObject* order = 0;
	PyObject* model = 0;
	int max_pcs = -1;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "ii|OOOOi", const_cast<char**>(kwlist),
		&rows, &cols, &input_mask, &output_mask, &order, &model, &max_pcs))
		return -1;

	if(order == Py_None)
		order = 0;

	if(order && !PyList_Check(order)) {
		PyErr_SetString(PyExc_TypeError, "Pixel order should be of type `list`.");
		return -1;
	}

	if(model == Py_None)
		model = 0;

	if(model && !PyType_IsSubtype(Py_TYPE(model), &MCBM_type)) {
		PyErr_SetString(PyExc_TypeError, "Model should be a subtype of `MCBM`.");
		return -1;
	}

	if(input_mask == Py_None)
		input_mask = 0;

	if(output_mask == Py_None)
		output_mask = 0;

	if(input_mask && output_mask) {
		input_mask = PyArray_FROM_OTF(input_mask, NPY_BOOL, NPY_F_CONTIGUOUS | NPY_ALIGNED);
		output_mask = PyArray_FROM_OTF(output_mask, NPY_BOOL, NPY_F_CONTIGUOUS | NPY_ALIGNED);

		if(!input_mask || !output_mask) {
			Py_XDECREF(input_mask);
			Py_XDECREF(output_mask);
			PyErr_SetString(PyExc_TypeError, "Masks have to be given as Boolean arrays.");
			return -1;
		}
	}

	// create the actual model
	try {
		if(order) {
			if(input_mask && output_mask) {
				self->patchMCBM = new PatchModel<MCBM, PCATransform>(
					rows,
					cols,
					PyList_AsTuples(order),
					PyArray_ToMatrixXb(input_mask),
					PyArray_ToMatrixXb(output_mask),
					model ? reinterpret_cast<MCBMObject*>(model)->mcbm : 0,
					max_pcs);

				Py_DECREF(input_mask);
				Py_DECREF(output_mask);
			} else {
				self->patchMCBM = new PatchModel<MCBM, PCATransform>(
					rows,
					cols,
					PyList_AsTuples(order),
					model ? reinterpret_cast<MCBMObject*>(model)->mcbm : 0,
					max_pcs);
			}
		} else {
			if(input_mask && output_mask) {
				self->patchMCBM = new PatchModel<MCBM, PCATransform>(
					rows,
					cols,
					PyArray_ToMatrixXb(input_mask),
					PyArray_ToMatrixXb(output_mask),
					model ? reinterpret_cast<MCBMObject*>(model)->mcbm : 0,
					max_pcs);

				Py_DECREF(input_mask);
				Py_DECREF(output_mask);
			} else {
				self->patchMCBM = new PatchModel<MCBM, PCATransform>(
					rows,
					cols,
					model ? reinterpret_cast<MCBMObject*>(model)->mcbm : 0,
					max_pcs);
			}
		}
	} catch(Exception exception) {
		Py_XDECREF(input_mask);
		Py_XDECREF(output_mask);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	return 0;
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
		PyErr_SetString(PyExc_TypeError, "Conditional distribution should be a subtype of `MCBM`.");
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

	if(self->patchMCBM->operator()(i, j).dimIn() != reinterpret_cast<MCBMObject*>(value)->mcbm->dimIn()) {
		PyErr_SetString(PyExc_ValueError, "Given model has wrong input dimensionality.");
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
	"Tries to guess reasonable parameters for all conditional distributions based on the data.\n"
	"\n"
	"It is assumed that the patches are stored in row-order ('C') in the columns of\n"
	"L{data}.\n"
	"\n"
	"@type  data: C{ndarray}\n"
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
		MCBM::Parameters* params = dynamic_cast<MCBM::Parameters*>(
			PyObject_ToMCBMParameters(parameters));

		self->patchMCBM->initialize(PyArray_ToMatrixXd(data), *params);

		delete params;

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
	"train(self, data, data_val=None, parameters=None)\n"
	"\n"
	"Trains the model to the given image patches by fitting each conditional\n"
	"distribution in turn.\n"
	"\n"
	"It is assumed that the patches are stored in row-order ('C') in the columns of\n"
	"L{data}. If hyperparameters are given, they are passed on to each conditional\n"
	"distribution.\n"
	"\n"
	"@type  data: C{ndarray}\n"
	"@param data: image patches stored column-wise\n"
	"\n"
	"@type  data_val: C{ndarray}\n"
	"@param data_val: image patches used for early stopping based on validation error\n"
	"\n"
	"@type  parameters: C{dict}\n"
	"@param parameters: a dictionary containing hyperparameters\n"
	"\n"
	"@rtype: C{bool}\n"
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

		MCBM::Parameters* params = dynamic_cast<MCBM::Parameters*>(
			PyObject_ToMCBMParameters(parameters));

		if(data_val) {
			if(i > -1 && j > -1)
				converged = self->patchMCBM->train(
					i, j,
					PyArray_ToMatrixXd(data),
					PyArray_ToMatrixXd(data_val),
					*params);
			else
				converged = self->patchMCBM->train(
					PyArray_ToMatrixXd(data),
					PyArray_ToMatrixXd(data_val),
					*params);
		} else {
			if(i > -1 && j > -1)
				converged = self->patchMCBM->train(
					i, j,
					PyArray_ToMatrixXd(data),
					*params);
			else
				converged = self->patchMCBM->train(
					PyArray_ToMatrixXd(data),
					*params);
		}

		delete params;

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



const char* PatchMCBM_reduce_doc =
	"__reduce__(self)\n"
	"\n"
	"Method used by Pickle.";

PyObject* PatchMCBM_reduce(PatchMCBMObject* self, PyObject*) {
	int rows = self->patchMCBM->rows();
	int cols = self->patchMCBM->cols();
	int maxPCs = self->patchMCBM->maxPCs();

	PyObject* order = PatchModel_order(
		reinterpret_cast<PatchModelObject*>(self), 0);
	PyObject* inputMask = PatchModel_input_mask(
		reinterpret_cast<PatchModelObject*>(self), 0);
	PyObject* outputMask = PatchModel_output_mask(
		reinterpret_cast<PatchModelObject*>(self), 0);

	// constructor arguments
	PyObject* args = Py_BuildValue("(iiOOOOi)",
		rows,
		cols,
		inputMask,
		outputMask,
		order,
		Py_None,
		maxPCs);
	
	Py_DECREF(order);
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

			if(PyErr_Occurred())
				return 0;
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
