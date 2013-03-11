#include "mcgsminterface.h"
#include "Eigen/Core"
#include "exception.h"
#include "callbacktrain.h"

using namespace Eigen;

MCGSM::Parameters PyObject_ToParameters(MCGSMObject* self, PyObject* parameters) {
	MCGSM::Parameters params;

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
	"p(c, s \\mid \\mathbf{x}) &\\propto \\exp\\left(\\eta_{cs} - \\frac{1}{2} e^{\\alpha_{cs}} \\sum_i \\beta_{ci}^2 \\left(\\mathbf{b}_i^\\top \\mathbf{x}\\right)^2 \\right),\\\\\n"
	"p(\\mathbf{y} \\mid c, s, \\mathbf{x}) &= |\\mathbf{L}_c| \\exp\\left(\\frac{M}{2}\\alpha_{cs} - \\frac{1}{2} e^{\\alpha_{cs}} (\\mathbf{y} - \\mathbf{A}_c \\mathbf{x})^\\top \\mathbf{L}_c \\mathbf{L}_c^\\top (\\mathbf{y} - \\mathbf{A}_c \\mathbf{x})\\right) / (2\\pi)^\\frac{M}{2}.\n"
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
	"\n"
	"which correspond to $\\eta_{cs}$, $\\alpha_{cs}$, $\\beta_{ci}$, $\\mathbf{b}_i$, "
	"$\\mathbf{L}_c$, and $\\mathbf{A}_c$, respectively.\n"
	"\n"
	"B{References:}\n"
	"\t- L. Theis, R. Hosseini, M. Bethge, I{Mixtures of Conditional Gaussian Scale "
	"Mixtures Applied to Multiscale Image Representations}, PLoS ONE, 2012.\n"
	"\n"
	"@type  dim_in: integer\n"
	"@param dim_in: dimensionality of input\n"
	"\n"
	"@type  dim_out: integer\n"
	"@param dim_out: dimensionality of output\n"
	"\n"
	"@type  num_components: integer\n"
	"@param num_components: number of components\n"
	"\n"
	"@type  num_scales: integer\n"
	"@param num_scales: number of scales per scale mixture component\n"
	"\n"
	"@type  num_features: integer\n"
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

	// create actual GSM instance
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



const char* MCGSM_initialize_doc =
	"initialize(self, input, output, parameters)\n"
	"\n"
	"Tries to guess more sensible initial values for the model parameters from data.\n"
	"\n"
	"@type  input: ndarray\n"
	"@param input: inputs stored in columns\n"
	"\n"
	"@type  output: ndarray\n"
	"@param output: outputs stored in columns\n"
	"\n"
	"@type  parameters: dict\n"
	"@param parameters: a dictionary containing hyperparameters";

PyObject* MCGSM_initialize(MCGSMObject* self, PyObject* args, PyObject* kwds) {
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
		Py_XDECREF(input);
		Py_XDECREF(output);
		PyErr_SetString(PyExc_TypeError, "Data has to be stored in NumPy arrays.");
		return 0;
	}

	try {
		self->mcgsm->initialize(
			PyArray_ToMatrixXd(input), 
			PyArray_ToMatrixXd(output), 
			PyObject_ToParameters(self, parameters));
		Py_DECREF(input);
		Py_DECREF(output);
		Py_INCREF(Py_None);
		return Py_None;
	} catch(Exception exception) {
		Py_DECREF(input);
		Py_DECREF(output);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	return 0;
}



const char* MCGSM_train_doc =
	"train(self, input, output, parameters=None)\n"
	"\n"
	"Fits model parameters to given data using L-BFGS.\n"
	"\n"
	"The following example demonstrates possible parameters and default settings.\n"
	"\n"
	"\t>>> model.train(input, output, parameters={\n"
	"\t>>> \t'verbosity': 0\n"
	"\t>>> \t'max_iter': 1000\n"
	"\t>>> \t'threshold': 1e-5\n"
	"\t>>> \t'num_grad': 20\n"
	"\t>>> \t'batch_size': 2000\n"
	"\t>>> \t'callback': None\n"
	"\t>>> \t'cb_iter': 25\n"
	"\t>>> \t'train_priors': True\n"
	"\t>>> \t'train_scales': True\n"
	"\t>>> \t'train_weights': True\n"
	"\t>>> \t'train_features': True\n"
	"\t>>> \t'train_cholesky_factors': True\n"
	"\t>>> \t'train_predictors': True\n"
	"\t>>> \t'regularize_features': 0.\n"
	"\t>>> \t'regularize_predictors': 0.\n"
	"\t>>> })\n"
	"\n"
	"The parameters C{train_priors}, C{train_scales}, and so on can be used to control which "
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
	"\t>>> def callback(i, mcgsm):\n"
	"\t>>> \tprint i\n"
	"\n"
	"@type  input: ndarray\n"
	"@param input: inputs stored in columns\n"
	"\n"
	"@type  output: ndarray\n"
	"@param output: outputs stored in columns\n"
	"\n"
	"@type  parameters: dict\n"
	"@param parameters: a dictionary containing hyperparameters\n"
	"\n"
	"@rtype: bool\n"
	"@return: C{True} if training converged, otherwise C{False}";

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
		Py_XDECREF(input);
		Py_XDECREF(output);
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



const char* MCGSM_posterior_doc =
	"posterior(self, input, output)\n"
	"\n"
	"Computes the posterior distribution over component labels, $p(c \\mid \\mathbf{x}, \\mathbf{y})$\n"
	"\n"
	"@type  input: ndarray\n"
	"@param input: inputs stored in columns\n"
	"\n"
	"@type  output: ndarray\n"
	"@param output: outputs stored in columns\n"
	"\n"
	"@rtype: ndarray\n"
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



const char* MCGSM_sample_posterior_doc =
	"sample_posterior(self, input, output)\n"
	"\n"
	"Samples component labels $c$ from the posterior $p(c \\mid \\mathbf{x}, \\mathbf{y})$.\n"
	"\n"
	"@type  input: ndarray\n"
	"@param input: inputs stored in columns\n"
	"\n"
	"@type  output: ndarray\n"
	"@param output: inputs stored in columns\n"
	"\n"
	"@rtype: ndarray\n"
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



const char* MCGSM_parameters_doc =
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



const char* MCGSM_set_parameters_doc =
	"set_parameters(self, x, parameters=None)\n"
	"\n"
	"Loads all model parameters from a vector as produced by L{parameters()}.\n"
	"\n"
	"@type  x: ndarray\n"
	"@param x: all model parameters concatenated to a vector\n"
	"\n"
	"@type  parameters: dict\n"
	"@param parameters: a dictionary containing hyperparameters";

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
		PyErr_SetString(PyExc_TypeError, "Data has to be stored in NumPy arrays.");
		return 0;
	}

	if(x)
		x = PyArray_FROM_OTF(x, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

	try {
		MCGSM::Parameters params = PyObject_ToParameters(self, parameters);

		MatrixXd gradient(self->mcgsm->numParameters(params), 1);

		if(x)
			self->mcgsm->computeGradient(
				PyArray_ToMatrixXd(input), 
				PyArray_ToMatrixXd(output), 
				PyArray_ToMatrixXd(x).data(), // TODO: PyArray_ToMatrixXd unnecessary
				gradient.data(), // TODO: don't use MatrixXd
				params);
		else
			self->mcgsm->computeGradient(
				PyArray_ToMatrixXd(input), 
				PyArray_ToMatrixXd(output), 
				self->mcgsm->parameters(params), // TODO: PyArray_ToMatrixXd unnecessary
				gradient.data(), // TODO: don't use MatrixXd
				params);

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
	PyObject* state = Py_BuildValue("(OOOOOO)", 
		priors, scales, weights, features, cholesky_factors, predictors);
	Py_DECREF(priors);
	Py_DECREF(scales);
	Py_DECREF(weights);
	Py_DECREF(features);
	Py_DECREF(cholesky_factors);
	Py_DECREF(predictors);

	PyObject* result = Py_BuildValue("(OOO)", self->ob_type, args, state);
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
