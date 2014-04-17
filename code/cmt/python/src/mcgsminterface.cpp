#include "callbackinterface.h"
#include "trainableinterface.h"
#include "conditionaldistributioninterface.h"
#include "preconditionerinterface.h"
#include "patchmodelinterface.h"
#include "mcgsminterface.h"

#include "Eigen/Core"
using Eigen::Map;

#include <map>
using std::pair;

#include "cmt/utils"
using CMT::Exception;

Trainable::Parameters* PyObject_ToMCGSMParameters(PyObject* parameters) {
	MCGSM::Parameters* params = dynamic_cast<MCGSM::Parameters*>(
		PyObject_ToParameters(parameters, new MCGSM::Parameters));

	// read parameters from dictionary
	if(parameters && parameters != Py_None) {
		PyObject* callback = PyDict_GetItemString(parameters, "callback");
		if(callback)
			if(PyCallable_Check(callback))
				params->callback = new CallbackInterface(&MCGSM_type, callback);
			else if(callback != Py_None)
				throw Exception("callback should be a function or callable object.");

		PyObject* train_priors = PyDict_GetItemString(parameters, "train_priors");
		if(train_priors)
			if(PyBool_Check(train_priors))
				params->trainPriors = (train_priors == Py_True);
			else
				throw Exception("train_priors should be of type `bool`.");

		PyObject* train_scales = PyDict_GetItemString(parameters, "train_scales");
		if(train_scales)
			if(PyBool_Check(train_scales))
				params->trainScales = (train_scales == Py_True);
			else
				throw Exception("train_scales should be of type `bool`.");

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

		PyObject* train_cholesky_factors = PyDict_GetItemString(parameters, "train_cholesky_factors");
		if(train_cholesky_factors)
			if(PyBool_Check(train_cholesky_factors))
				params->trainCholeskyFactors = (train_cholesky_factors == Py_True);
			else
				throw Exception("train_cholesky_factors should be of type `bool`.");

		PyObject* train_predictors = PyDict_GetItemString(parameters, "train_predictors");
		if(train_predictors)
			if(PyBool_Check(train_predictors))
				params->trainPredictors = (train_predictors == Py_True);
			else
				throw Exception("train_predictors should be of type `bool`.");

		PyObject* train_linear_features = PyDict_GetItemString(parameters, "train_linear_features");
		if(train_linear_features)
			if(PyBool_Check(train_linear_features))
				params->trainLinearFeatures = (train_linear_features == Py_True);
			else
				throw Exception("train_linear_features should be of type `bool`.");

		PyObject* train_means = PyDict_GetItemString(parameters, "train_means");
		if(train_means)
			if(PyBool_Check(train_means))
				params->trainMeans = (train_means == Py_True);
			else
				throw Exception("train_means should be of type `bool`.");

		PyObject* regularize_features = PyDict_GetItemString(parameters, "regularize_features");
		if(regularize_features)
			params->regularizeFeatures = PyObject_ToRegularizer(regularize_features);

		PyObject* regularize_predictors = PyDict_GetItemString(parameters, "regularize_predictors");
		if(regularize_predictors)
			params->regularizePredictors = PyObject_ToRegularizer(regularize_predictors);

		PyObject* regularize_weights = PyDict_GetItemString(parameters, "regularize_weights");
		if(regularize_weights)
			params->regularizeWeights = PyObject_ToRegularizer(regularize_weights);

		PyObject* regularize_linear_features = PyDict_GetItemString(parameters, "regularize_linear_features");
		if(regularize_linear_features)
			params->regularizeLinearFeatures = PyObject_ToRegularizer(regularize_linear_features);

		PyObject* regularize_means = PyDict_GetItemString(parameters, "regularize_means");
		if(regularize_means)
			params->regularizeMeans = PyObject_ToRegularizer(regularize_means);
	}

	return params;
}


const char* MCGSM_doc =
	"An implementation of a mixture of conditional Gaussian scale mixtures.\n"
	"\n"
	"The distribution defined by the model is\n"
	"\n"
	"$$p(\\mathbf{y} \\mid \\mathbf{x}) = \\sum_{c, s} p(c, s \\mid \\mathbf{x}) p(\\mathbf{y} \\mid c, s, \\mathbf{x}),$$\n"
	"\n"
	"where\n"
	"\n"
	"\\begin{align}\n"
	"p(c, s \\mid \\mathbf{x}) &\\propto \\exp\\left(\\eta_{cs} - \\frac{1}{2} e^{\\alpha_{cs}} \\sum_i \\beta_{ci}^2 \\left(\\mathbf{b}_i^\\top \\mathbf{x}\\right)^2 + e^{\\alpha_{cs}} \\mathbf{w}_c^\\top \\mathbf{x} \\right),\\\\\n"
	"p(\\mathbf{y} \\mid c, s, \\mathbf{x}) &= |\\mathbf{L}_c| \\exp\\left(\\frac{M}{2}\\alpha_{cs} - \\frac{1}{2} e^{\\alpha_{cs}} (\\mathbf{y} - \\mathbf{A}_c \\mathbf{x} - \\mathbf{u}_c)^\\top \\mathbf{L}_c \\mathbf{L}_c^\\top (\\mathbf{y} - \\mathbf{A}_c \\mathbf{x} - \\mathbf{u}_c)\\right) / (2\\pi)^\\frac{M}{2}.\n"
	"\\end{align}\n"
	"\n"
	"To create an MCGSM for $N$-dimensional inputs $\\mathbf{x} \\in \\mathbb{R}^N$ "
	"and $M$-dimensional outputs $\\mathbf{y} \\in \\mathbb{R}^M$ with, for example, 8 predictors "
	"$\\mathbf{A}_c$, 6 scales $\\alpha_{cs}$ per component $c$, and 100 features $\\mathbf{b}_i$, use\n"
	"\n"
	"\t>>> mcgsm = MCGSM(N, M, 8, 6, 100)\n"
	"\n"
	"To access the different parameters, you can use\n"
	"\n"
	"\t>>> mcgsm.priors\n"
	"\t>>> mcgsm.scales\n"
	"\t>>> mcgsm.weights\n"
	"\t>>> mcgsm.features\n"
	"\t>>> mcgsm.cholesky_factors\n"
	"\t>>> mcgsm.predictors\n"
	"\t>>> mcgsm.linear_features\n"
	"\t>>> mcgsm.means\n"
	"\n"
	"which correspond to $\\eta_{cs}$, $\\alpha_{cs}$, $\\beta_{ci}$, $\\mathbf{b}_i$, "
	"$\\mathbf{L}_c$, and $\\mathbf{A}_c$, respectively.\n"
	"\n"
	"B{References:}\n"
	"\t- L. Theis, R. Hosseini, M. Bethge, I{Mixtures of Conditional Gaussian Scale "
	"Mixtures Applied to Multiscale Image Representations}, PLoS ONE, 2012.\n"
	"\n"
	"@type  dim_in: C{int}\n"
	"@param dim_in: dimensionality of input\n"
	"\n"
	"@type  dim_out: C{int}\n"
	"@param dim_out: dimensionality of output\n"
	"\n"
	"@type  num_components: C{int}\n"
	"@param num_components: number of components\n"
	"\n"
	"@type  num_scales: C{int}\n"
	"@param num_scales: number of scales per scale mixture component\n"
	"\n"
	"@type  num_features: C{int}\n"
	"@param num_features: number of features used to approximate input covariance matrices";

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

	// create actual MCGSM instance
	try {
		self->mcgsm = new MCGSM(dim_in, dim_out, num_components, num_scales, num_features);
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	return 0;
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



PyObject* MCGSM_linear_features(MCGSMObject* self, PyObject*, void*) {
	PyObject* array = PyArray_FromMatrixXd(self->mcgsm->linearFeatures());

	// make array immutable
	reinterpret_cast<PyArrayObject*>(array)->flags &= ~NPY_WRITEABLE;

	return array;
}



int MCGSM_set_linear_features(MCGSMObject* self, PyObject* value, void*) {
	value = PyArray_FROM_OTF(value, NPY_DOUBLE, NPY_IN_ARRAY);

	if(!value) {
		PyErr_SetString(PyExc_TypeError, "Linear features should be of type `ndarray`.");
		return -1;
	}

	try {
		self->mcgsm->setLinearFeatures(PyArray_ToMatrixXd(value));
	} catch(Exception exception) {
		Py_DECREF(value);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	Py_DECREF(value);

	return 0;
}



PyObject* MCGSM_means(MCGSMObject* self, PyObject*, void*) {
	PyObject* array = PyArray_FromMatrixXd(self->mcgsm->means());

	// make array immutable
	reinterpret_cast<PyArrayObject*>(array)->flags &= ~NPY_WRITEABLE;

	return array;
}



int MCGSM_set_means(MCGSMObject* self, PyObject* value, void*) {
	value = PyArray_FROM_OTF(value, NPY_DOUBLE, NPY_IN_ARRAY);

	if(!value) {
		PyErr_SetString(PyExc_TypeError, "Means should be of type `ndarray`.");
		return -1;
	}

	try {
		self->mcgsm->setMeans(PyArray_ToMatrixXd(value));
	} catch(Exception exception) {
		Py_DECREF(value);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	Py_DECREF(value);

	return 0;
}



const char* MCGSM_train_doc =
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
	"\t>>> \t'train_scales': True,\n"
	"\t>>> \t'train_weights': True,\n"
	"\t>>> \t'train_features': True,\n"
	"\t>>> \t'train_cholesky_factors': True,\n"
	"\t>>> \t'train_predictors': True,\n"
	"\t>>> \t'train_linear_features': False,\n"
	"\t>>> \t'train_means': False,\n"
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
	"\t>>> \t'regularize_linear_features': {\n"
	"\t>>> \t\t'strength': 0.,\n"
	"\t>>> \t\t'transform': None,\n"
	"\t>>> \t\t'norm': 'L2'},\n"
	"\t>>> \t'regularize_means': {\n"
	"\t>>> \t\t'strength': 0.,\n"
	"\t>>> \t\t'transform': None,\n"
	"\t>>> \t\t'norm': 'L2'},\n"
	"\t>>> })\n"
	"\n"
	"The parameters C{train_priors}, C{train_scales}, and so on can be used to control which "
	"parameters will be optimized. Optimization stops after C{max_iter} iterations or if "
	"the gradient is sufficiently small enough, as specified by C{threshold}."
	"C{num_grad} is the number of gradients used by L-BFGS to approximate the inverse Hessian "
	"matrix.\n"
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
	"\t>>> def callback(i, mcgsm):\n"
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

PyObject* MCGSM_train(MCGSMObject* self, PyObject* args, PyObject* kwds) {
	return Trainable_train(
		reinterpret_cast<TrainableObject*>(self), 
		args, 
		kwds,
		&PyObject_ToMCGSMParameters);
}



PyObject* MCGSM_check_performance(MCGSMObject* self, PyObject* args, PyObject* kwds) {
	return Trainable_check_performance(
		reinterpret_cast<TrainableObject*>(self), 
		args, 
		kwds,
		&PyObject_ToMCGSMParameters);
}



PyObject* MCGSM_check_gradient(MCGSMObject* self, PyObject* args, PyObject* kwds) {
	return Trainable_check_gradient(
		reinterpret_cast<TrainableObject*>(self), 
		args, 
		kwds,
		&PyObject_ToMCGSMParameters);
}



const char* MCGSM_loglikelihood_doc =
	"loglikelihood(self, input, output, labels=None)\n"
	"\n"
	"Computes the conditional log-likelihood for the given data points in nats.\n"
	"If labels are specified, the log-likelihood of the corresponding mixture\n"
	"component is computed.\n"
	"\n"
	"@type  input: ndarray\n"
	"@param input: inputs stored in columns\n"
	"\n"
	"@type  output: ndarray\n"
	"@param output: outputs stored in columns\n"
	"\n"
	"@type  labels: ndarray\n"
	"@param labels: indices indicating mixture components\n"
	"\n"
	"@rtype: ndarray\n"
	"@return: log-likelihood of the model evaluated for each data point";

PyObject* MCGSM_loglikelihood(MCGSMObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"input", "output", "labels", 0};

	PyObject* input;
	PyObject* output;
	PyObject* labels = 0;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "OO|O",
		const_cast<char**>(kwlist), &input, &output, &labels))
		return 0;

	// make sure data is stored in NumPy array
	input = PyArray_FROM_OTF(input, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
	output = PyArray_FROM_OTF(output, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

	if(!input || !output) {
		PyErr_SetString(PyExc_TypeError, "Data has to be stored in NumPy arrays.");
		return 0;
	}

	if(labels == Py_None)
		labels = 0;

	if(labels) {
		labels = PyArray_FROM_OTF(labels, NPY_INT64, NPY_F_CONTIGUOUS | NPY_ALIGNED);

		if(!labels) {
			PyErr_SetString(PyExc_TypeError, "Labels have to be stored in an integer NumPy array.");
			return 0;
		} else if(PyArray_DIM(labels, 0) > 1) {
			PyErr_SetString(PyExc_TypeError, "Labels have to be stored in one row.");
			return 0;
		}
	}

	try {
		PyObject* result;
		if(labels)
			result = PyArray_FromMatrixXd(
				self->mcgsm->logLikelihood(
					PyArray_ToMatrixXd(input),
					PyArray_ToMatrixXd(output),
					PyArray_ToMatrixXi(labels)));
		else
			result = PyArray_FromMatrixXd(
				self->mcgsm->logLikelihood(
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



const char* MCGSM_sample_doc =
	"sample(self, input, labels=None)\n"
	"\n"
	"Generates outputs for given inputs.\n"
	"If labels are specified, uses the given mixture component to generate outputs.\n"
	"\n"
	"@type  input: ndarray\n"
	"@param input: inputs stored in columns\n"
	"\n"
	"@type  labels: ndarray\n"
	"@param labels: indices indicating mixture components\n"
	"\n"
	"@rtype: ndarray\n"
	"@return: sampled outputs";

PyObject* MCGSM_sample(MCGSMObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"input", "labels", 0};

	PyObject* input;
	PyObject* labels = 0;

	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O|O", 
		const_cast<char**>(kwlist), &input, &labels))
		return 0;

	input = PyArray_FROM_OTF(input, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

	if(!input) {
		PyErr_SetString(PyExc_TypeError, "Data has to be stored in a NumPy array.");
		return 0;
	}

	if(labels == Py_None)
		labels = 0;

	if(labels) {
		labels = PyArray_FROM_OTF(labels, NPY_INT64, NPY_F_CONTIGUOUS | NPY_ALIGNED);

		if(!labels) {
			PyErr_SetString(PyExc_TypeError, "Labels have to be stored in an integer NumPy array.");
			return 0;
		}

		if(PyArray_DIM(labels, 0) > 1) {
			PyErr_SetString(PyExc_TypeError, "Labels have to be stored in one row.");
			return 0;
		}
	}

	try {
		PyObject* result;
		if(labels)
			result = PyArray_FromMatrixXd(
				self->mcgsm->sample(
					PyArray_ToMatrixXd(input),
					PyArray_ToMatrixXi(labels)));
		else
			result = PyArray_FromMatrixXd(
				self->mcgsm->sample(PyArray_ToMatrixXd(input)));
		Py_DECREF(input);
		return result;
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		Py_DECREF(input);
		return 0;
	}

	return 0;
}



const char* MCGSM_prior_doc =
	"prior(self, input)\n"
	"\n"
	"Computes the prior distribution over component labels, $p(c \\mid \\mathbf{x})$\n"
	"\n"
	"@type  input: C{ndarray}\n"
	"@param input: inputs stored in columns\n"
	"\n"
	"@rtype: C{ndarray}\n"
	"@return: a distribution over labels for each given input";

PyObject* MCGSM_prior(MCGSMObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"input", 0};

	PyObject* input;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O", const_cast<char**>(kwlist), &input))
		return 0;

	// make sure data is stored in NumPy array
	input = PyArray_FROM_OTF(input, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

	if(!input) {
		PyErr_SetString(PyExc_TypeError, "Data has to be stored in NumPy arrays.");
		return 0;
	}

	try {
		PyObject* result = PyArray_FromMatrixXd(
			self->mcgsm->prior(PyArray_ToMatrixXd(input)));
		Py_DECREF(input);
		return result;
	} catch(Exception exception) {
		Py_DECREF(input);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	return 0;
}



const char* MCGSM_posterior_doc =
	"posterior(self, input, output)\n"
	"\n"
	"Computes the posterior distribution over component labels, $p(c \\mid \\mathbf{x}, \\mathbf{y})$\n"
	"\n"
	"@type  input: C{ndarray}\n"
	"@param input: inputs stored in columns\n"
	"\n"
	"@type  output: C{ndarray}\n"
	"@param output: outputs stored in columns\n"
	"\n"
	"@rtype: C{ndarray}\n"
	"@return: a posterior distribution over labels for each given pair of input and output";

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



const char* MCGSM_sample_prior_doc =
	"sample_prior(self, input)\n"
	"\n"
	"Samples component labels $c$ from the distribution $p(c \\mid \\mathbf{x})$.\n"
	"\n"
	"@type  input: C{ndarray}\n"
	"@param input: inputs stored in columns\n"
	"\n"
	"@rtype: C{ndarray}\n"
	"@return: an integer array containing a sampled index for each input and output pair";

PyObject* MCGSM_sample_prior(MCGSMObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"input", 0};

	PyObject* input;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O", const_cast<char**>(kwlist), &input))
		return 0;

	// make sure data is stored in NumPy array
	input = PyArray_FROM_OTF(input, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

	if(!input) {
		Py_XDECREF(input);
		PyErr_SetString(PyExc_TypeError, "Data has to be stored in NumPy arrays.");
		return 0;
	}

	try {
		PyObject* result = PyArray_FromMatrixXi(
			self->mcgsm->samplePrior(PyArray_ToMatrixXd(input)));
		Py_DECREF(input);
		return result;
	} catch(Exception exception) {
		Py_DECREF(input);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	return 0;
}



const char* MCGSM_sample_posterior_doc =
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
		Py_XDECREF(input);
		Py_XDECREF(output);
		PyErr_SetString(PyExc_TypeError, "Data has to be stored in NumPy arrays.");
		return 0;
	}

	try {
		PyObject* result = PyArray_FromMatrixXi(
			self->mcgsm->samplePosterior(
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



PyObject* MCGSM_parameters(MCGSMObject* self, PyObject* args, PyObject* kwds) {
	return Trainable_parameters(
		reinterpret_cast<TrainableObject*>(self), 
		args, 
		kwds,
		&PyObject_ToMCGSMParameters);
}



PyObject* MCGSM_set_parameters(MCGSMObject* self, PyObject* args, PyObject* kwds) {
	return Trainable_set_parameters(
		reinterpret_cast<TrainableObject*>(self), 
		args, 
		kwds,
		&PyObject_ToMCGSMParameters);
}



PyObject* MCGSM_parameter_gradient(MCGSMObject* self, PyObject* args, PyObject* kwds) {
	return Trainable_parameter_gradient(
		reinterpret_cast<TrainableObject*>(self), 
		args, 
		kwds,
		&PyObject_ToMCGSMParameters);
}



PyObject* MCGSM_compute_data_gradient(MCGSMObject* self, PyObject* args, PyObject* kwds) {
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
		pair<pair<ArrayXXd, ArrayXXd>, Array<double, 1, Dynamic> > gradients =
			 self->mcgsm->computeDataGradient(
				PyArray_ToMatrixXd(input), 
				PyArray_ToMatrixXd(output));

		PyObject* inputGradient = PyArray_FromMatrixXd(gradients.first.first);
		PyObject* outputGradient = PyArray_FromMatrixXd(gradients.first.second);
		PyObject* logLikelihood = PyArray_FromMatrixXd(gradients.second);
		PyObject* tuple = Py_BuildValue("(OOO)", inputGradient, outputGradient, logLikelihood);

		Py_DECREF(inputGradient);
		Py_DECREF(outputGradient);
		Py_DECREF(logLikelihood);

		Py_DECREF(input);
		Py_DECREF(output);

		return tuple;
	} catch(Exception exception) {
		Py_DECREF(input);
		Py_DECREF(output);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}
}



const char* MCGSM_reduce_doc =
	"__reduce__(self)\n"
	"\n"
	"Method used by Pickle.";

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
	PyObject* linear_features = MCGSM_linear_features(self, 0, 0);
	PyObject* means = MCGSM_means(self, 0, 0);
	PyObject* state = Py_BuildValue("(OOOOOOOO)", 
		priors, scales, weights, features, cholesky_factors, predictors, linear_features, means);
	Py_DECREF(priors);
	Py_DECREF(scales);
	Py_DECREF(weights);
	Py_DECREF(features);
	Py_DECREF(cholesky_factors);
	Py_DECREF(predictors);
	Py_DECREF(linear_features);
	Py_DECREF(means);

	PyObject* result = Py_BuildValue("(OOO)", Py_TYPE(self), args, state);
	Py_DECREF(args);
	Py_DECREF(state);

	return result;
}



const char* MCGSM_setstate_doc =
	"__setstate__(self)\n"
	"\n"
	"Method used by Pickle.";

PyObject* MCGSM_setstate(MCGSMObject* self, PyObject* state, PyObject*) {
	PyObject* priors;
	PyObject* scales;
	PyObject* weights;
	PyObject* features;
	PyObject* cholesky_factors;
	PyObject* predictors;
	PyObject* linear_features = 0;
	PyObject* means = 0;

	if(!PyArg_ParseTuple(state, "(OOOOOOOO)", &priors, &scales, &weights, &features, &cholesky_factors, &predictors, &linear_features, &means)) {
		PyErr_Clear();

		// try without means for backwards-compatibility reasons
		if(!PyArg_ParseTuple(state, "(OOOOOO)", &priors, &scales, &weights, &features, &cholesky_factors, &predictors))
			return 0;
	}

	try {
		MCGSM_set_priors(self, priors, 0);
		MCGSM_set_scales(self, scales, 0);
		MCGSM_set_weights(self, weights, 0);
		MCGSM_set_features(self, features, 0);
		MCGSM_set_cholesky_factors(self, cholesky_factors, 0);
		MCGSM_set_predictors(self, predictors, 0);
		if(linear_features && means) {
			MCGSM_set_linear_features(self, linear_features, 0);
			MCGSM_set_means(self, means, 0);
		}
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	Py_INCREF(Py_None);
	return Py_None;
}



const char* PatchMCGSM_doc =
	"Model image patches by using an L{MCGSM} for each conditional distribution.\n"
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
	"@type  model: L{MCGSM}\n"
	"@param model: model used as a template to initialize all conditional distributions\n"
	"\n"
	"@type  max_pcs: C{int}\n"
	"@param max_pcs: can be used to reduce dimensionality of inputs to conditional models";

int PatchMCGSM_init(PatchMCGSMObject* self, PyObject* args, PyObject* kwds) {
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

	if(model && !PyType_IsSubtype(Py_TYPE(model), &MCGSM_type)) {
		PyErr_SetString(PyExc_TypeError, "Model should be a subtype of `MCGSM`.");
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
				self->patchMCGSM = new PatchModel<MCGSM, PCAPreconditioner>(
					rows,
					cols,
					PyList_AsTuples(order),
					PyArray_ToMatrixXb(input_mask),
					PyArray_ToMatrixXb(output_mask),
					model ? reinterpret_cast<MCGSMObject*>(model)->mcgsm : 0,
					max_pcs);

				Py_DECREF(input_mask);
				Py_DECREF(output_mask);
			} else {
				self->patchMCGSM = new PatchModel<MCGSM, PCAPreconditioner>(
					rows,
					cols,
					PyList_AsTuples(order),
					model ? reinterpret_cast<MCGSMObject*>(model)->mcgsm : 0,
					max_pcs);
			}
		} else {
			if(input_mask && output_mask) {
				self->patchMCGSM = new PatchModel<MCGSM, PCAPreconditioner>(
					rows,
					cols,
					PyArray_ToMatrixXb(input_mask),
					PyArray_ToMatrixXb(output_mask),
					model ? reinterpret_cast<MCGSMObject*>(model)->mcgsm : 0,
					max_pcs);

				Py_DECREF(input_mask);
				Py_DECREF(output_mask);
			} else {
				self->patchMCGSM = new PatchModel<MCGSM, PCAPreconditioner>(
					rows,
					cols,
					model ? reinterpret_cast<MCGSMObject*>(model)->mcgsm : 0,
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



PyObject* PatchMCGSM_subscript(PatchMCGSMObject* self, PyObject* key) {
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

	PyObject* obj = CD_new(&MCGSM_type, 0, 0);
	reinterpret_cast<MCGSMObject*>(obj)->mcgsm = &self->patchMCGSM->operator()(i, j);
	reinterpret_cast<MCGSMObject*>(obj)->owner = false;
	Py_INCREF(obj);

	return obj;
}



int PatchMCGSM_ass_subscript(PatchMCGSMObject* self, PyObject* key, PyObject* value) {
	if(!PyType_IsSubtype(Py_TYPE(value), &MCGSM_type)) {
		PyErr_SetString(PyExc_TypeError, "Conditional distribution should be a subtype of `MCGSM`.");
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

	if(self->patchMCGSM->operator()(i, j).dimIn() != reinterpret_cast<MCGSMObject*>(value)->mcgsm->dimIn()) {
		PyErr_SetString(PyExc_ValueError, "Given model has wrong input dimensionality.");
		return -1;
	}

	self->patchMCGSM->operator()(i, j) = *reinterpret_cast<MCGSMObject*>(value)->mcgsm;

	return 0;
}



PyObject* PatchMCGSM_preconditioner(PatchMCGSMObject* self, PyObject* args) {
	int i;
	int j;

	if(!PyArg_ParseTuple(args, "ii", &i, &j)) {
		PyErr_SetString(PyExc_TypeError, "Index should consist of a row and a column.");
		return 0;
	}

	try {
		PCAPreconditioner* pc = &self->patchMCGSM->preconditioner(i, j);
		PyObject* preconditioner = Preconditioner_new(&PCAPreconditioner_type, 0, 0);
		reinterpret_cast<PCAPreconditionerObject*>(preconditioner)->owner = false;
		reinterpret_cast<PCAPreconditionerObject*>(preconditioner)->preconditioner = pc;
		Py_INCREF(preconditioner);
		return preconditioner;
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	return 0;
}



PyObject* PatchMCGSM_preconditioners(PatchMCGSMObject* self, void*) {
	if(self->patchMCGSM->maxPCs() < 0)
		return PyDict_New();

	PyObject* preconditioners = PyDict_New();

	for(int i = 0; i < self->patchMCGSM->rows(); ++i) {
		for(int j = 0; j < self->patchMCGSM->cols(); ++j) {
			PyObject* index = Py_BuildValue("(ii)", i, j);
			PyObject* preconditioner = PatchMCGSM_preconditioner(self, index);

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



int PatchMCGSM_set_preconditioners(PatchMCGSMObject* self, PyObject* value, void*) {
	if(!PyDict_Check(value)) {
		PyErr_SetString(PyExc_RuntimeError, "Preconditioners have to be stored in a dictionary."); 
		return -1;
	}

	for(int i = 0; i < self->patchMCGSM->rows(); ++i)
		for(int j = 0; j < self->patchMCGSM->cols(); ++j) {
			PyObject* index = Py_BuildValue("(ii)", i, j);
			PyObject* preconditioner = PyDict_GetItem(value, index);

			if(!preconditioner)
				continue;

			if(!PyType_IsSubtype(Py_TYPE(preconditioner), &PCAPreconditioner_type)) {
				PyErr_SetString(PyExc_RuntimeError,
					"All preconditioners must be of type `PCAPreconditioner`.");
				return -1;
			}

			try {
 				self->patchMCGSM->setPreconditioner(i, j,
 					*reinterpret_cast<PCAPreconditionerObject*>(preconditioner)->preconditioner);
			} catch(Exception exception) {
				PyErr_SetString(PyExc_RuntimeError, exception.message());
				return -1;
			}

			Py_DECREF(index);
		}

	return 0;
}



const char* PatchMCGSM_initialize_doc =
	"initialize(self, data, parameters=None)\n"
	"\n"
	"Tries to guess reasonable parameters for all conditional distributions based on the data.\n"
	"\n"
	"It is assumed that the patches are stored in row-order ('C') in the columns of\n"
	"L{data}.\n"
	"\n"
	"@type  data: C{ndarray}\n"
	"@param data: image patches stored column-wise";

PyObject* PatchMCGSM_initialize(PatchMCGSMObject* self, PyObject* args, PyObject* kwds) {
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
		MCGSM::Parameters* params = dynamic_cast<MCGSM::Parameters*>(
			PyObject_ToMCGSMParameters(parameters));

		self->patchMCGSM->initialize(PyArray_ToMatrixXd(data), *params);

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



const char* PatchMCGSM_train_doc =
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

PyObject* PatchMCGSM_train(PatchMCGSMObject* self, PyObject* args, PyObject* kwds) {
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

		MCGSM::Parameters* params = dynamic_cast<MCGSM::Parameters*>(
			PyObject_ToMCGSMParameters(parameters));

		if(data_val) {
			if(i > -1 && j > -1)
				converged = self->patchMCGSM->train(
					i, j,
					PyArray_ToMatrixXd(data),
					PyArray_ToMatrixXd(data_val),
					*params);
			else
				converged = self->patchMCGSM->train(
					PyArray_ToMatrixXd(data),
					PyArray_ToMatrixXd(data_val),
					*params);
		} else {
			if(i > -1 && j > -1)
				converged = self->patchMCGSM->train(
					i, j,
					PyArray_ToMatrixXd(data),
					*params);
			else
				converged = self->patchMCGSM->train(
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



const char* PatchMCGSM_reduce_doc =
	"__reduce__(self)\n"
	"\n"
	"Method used by Pickle.";

PyObject* PatchMCGSM_reduce(PatchMCGSMObject* self, PyObject*) {
	int rows = self->patchMCGSM->rows();
	int cols = self->patchMCGSM->cols();
	int maxPCs = self->patchMCGSM->maxPCs();

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
	PyObject* models = PyTuple_New(self->patchMCGSM->dim());

	for(int i = 0; i < rows; ++i)
		for(int j = 0; j < cols; ++j) {
			PyObject* index = Py_BuildValue("(ii)", i, j);
			PyObject* mcbm = PatchMCGSM_subscript(self, index);

			// add MCGSM to list of models
			PyTuple_SetItem(models, i * cols + j, mcbm);

			Py_DECREF(index);
		}

	PyObject* preconditioners = PatchMCGSM_preconditioners(self, 0);

	PyObject* state = Py_BuildValue("(OO)", models, preconditioners);
	PyObject* result = Py_BuildValue("(OOO)", Py_TYPE(self), args, state);

	Py_DECREF(preconditioners);
	Py_DECREF(args);
	Py_DECREF(state);

	return result;
}



const char* PatchMCGSM_setstate_doc =
	"__setstate__(self)\n"
	"\n"
	"Method used by Pickle.";

PyObject* PatchMCGSM_setstate(PatchMCGSMObject* self, PyObject* state) {
	int cols = self->patchMCGSM->cols();

	// for some reason the actual state is encapsulated in another tuple
	state = PyTuple_GetItem(state, 0);

	PyObject* models = PyTuple_GetItem(state, 0);
  	PyObject* preconditioners = PyTuple_GetItem(state, 1);

	if(PyTuple_Size(models) != self->patchMCGSM->dim()) {
		PyErr_SetString(PyExc_RuntimeError, "Something went wrong while unpickling the model.");
		return 0;
	}

	try {
		for(int i = 0; i < self->patchMCGSM->dim(); ++i) {
			PyObject* index = Py_BuildValue("(ii)", i / cols, i % cols);
			PatchMCGSM_ass_subscript(self, index, PyTuple_GetItem(models, i));
			Py_DECREF(index);

			if(PyErr_Occurred())
				return 0;
		}
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

  	if(preconditioners)
  		PatchMCGSM_set_preconditioners(self, preconditioners, 0);

	Py_INCREF(Py_None);
	return Py_None;
}
