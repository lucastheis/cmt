#include "distributioninterface.h"
#include "mixtureinterface.h"
#include "pyutils.h"
#include "gsminterface.h"

#include "cmt/utils"
using CMT::Exception;

const char* Mixture_doc =
	"Basic mixture class implementing expectation maximization.\n"
	"\n"
	"$$p(\\mathbf{x}) = \\sum_k \\pi_k p_k(\\mathbf{x})$$\n"
	"\n"
	"To access the parameters $\\pi_k$, use C{mixture.priors}. Mixture components are added\n"
	"manually via C{mixture.add_component(component)}. To access component $k$, you can use\n"
	"C{mixture[k]}.\n"
	"\n"
	"@type  dim: C{int}\n"
	"@param dim: dimensionality of the data";

int Mixture_init(MixtureObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"dim", 0};

	int dim;

	if(!PyArg_ParseTupleAndKeywords(args, kwds, "i", const_cast<char**>(kwlist), &dim))
		return -1;

	try {
		self->mixture = new Mixture(dim);
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	return 0;
}


const char* MoGSM_doc =
	"Mixture of L{Gaussian scale mixtures<models.GSM>}.\n"
	"\n"
	"@type  dim: C{int}\n"
	"@param dim: dimensionality of the data\n"
	"\n"
	"@type  num_components: C{int}\n"
	"@param num_components: number of initial mixture components\n"
	"\n"
	"@type  num_scales: C{int}\n"
	"@param num_scales: number of scales per mixture component";

int MoGSM_init(MoGSMObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"dim", "num_components", "num_scales", 0};

	int dim;
	int num_components;
	int num_scales = 6;

	if(!PyArg_ParseTupleAndKeywords(args, kwds, "ii|i", const_cast<char**>(kwlist),
		&dim, &num_components, &num_scales))
		return -1;

	try {
		self->mixture = new MoGSM(dim, num_components, num_scales);

		for(int k = 0; k < num_components; ++k)
			self->componentTypes.push_back(&GSM_type);
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	return 0;
}



PyObject* Mixture_num_components(MixtureObject* self, void*) {
	return PyInt_FromLong(self->mixture->numComponents());
}



PyObject* MoGSM_num_scales(MoGSMObject* self, void*) {
	return PyInt_FromLong(self->mixture->numScales());
}



Mixture::Component::Parameters* PyObject_ToMixtureComponentParameters(PyObject* parameters) {
	Mixture::Component::Parameters* params = new Mixture::Component::Parameters;

	if(parameters && parameters != Py_None) {
		PyObject* verbosity = PyDict_GetItemString(parameters, "verbosity");
		if(verbosity)
			if(PyInt_Check(verbosity))
				params->verbosity = PyInt_AsLong(verbosity);
			else if(PyFloat_Check(verbosity))
				params->verbosity = static_cast<int>(PyFloat_AsDouble(verbosity));
			else
				throw Exception("verbosity should be of type `int`.");

		PyObject* max_iter = PyDict_GetItemString(parameters, "max_iter");
		if(max_iter)
			if(PyInt_Check(max_iter))
				params->maxIter = PyInt_AsLong(max_iter);
			else if(PyFloat_Check(max_iter))
				params->maxIter = static_cast<int>(PyFloat_AsDouble(max_iter));
			else
				throw Exception("max_iter should be of type `int`.");

		PyObject* threshold = PyDict_GetItemString(parameters, "threshold");
		if(threshold)
			if(PyFloat_Check(threshold))
				params->threshold = PyFloat_AsDouble(threshold);
			else if(PyInt_Check(threshold))
				params->threshold = static_cast<double>(PyFloat_AsDouble(threshold));
			else
				throw Exception("threshold should be of type `float`.");

		PyObject* train_priors = PyDict_GetItemString(parameters, "train_priors");
		if(train_priors)
			if(PyBool_Check(train_priors))
				params->trainPriors = (train_priors == Py_True);
			else
				throw Exception("train_priors should be of type `bool`.");

		PyObject* train_covariance = PyDict_GetItemString(parameters, "train_covariance");
		if(train_covariance)
			if(PyBool_Check(train_covariance))
				params->trainCovariance = (train_covariance == Py_True);
			else
				throw Exception("train_covariance should be of type `bool`.");

		PyObject* train_scales = PyDict_GetItemString(parameters, "train_scales");
		if(train_scales)
			if(PyBool_Check(train_scales))
				params->trainScales = (train_scales == Py_True);
			else
				throw Exception("train_scales should be of type `bool`.");

		PyObject* train_mean = PyDict_GetItemString(parameters, "train_mean");
		if(train_mean)
			if(PyBool_Check(train_mean))
				params->trainMean = (train_mean == Py_True);
			else
				throw Exception("train_mean should be of type `bool`.");

		PyObject* regularize_priors = PyDict_GetItemString(parameters, "regularize_priors");
		if(regularize_priors)
			if(PyFloat_Check(regularize_priors))
				params->regularizePriors = PyFloat_AsDouble(regularize_priors);
			else if(PyInt_Check(regularize_priors))
				params->regularizePriors = static_cast<double>(PyFloat_AsDouble(regularize_priors));
			else
				throw Exception("regularize_priors should be of type `float`.");

		PyObject* regularize_covariance = PyDict_GetItemString(parameters, "regularize_covariance");
		if(regularize_covariance)
			if(PyFloat_Check(regularize_covariance))
				params->regularizeCovariance = PyFloat_AsDouble(regularize_covariance);
			else if(PyInt_Check(regularize_covariance))
				params->regularizeCovariance = static_cast<double>(PyFloat_AsDouble(regularize_covariance));
			else
				throw Exception("regularize_covariance should be of type `float`.");

		PyObject* regularize_scales = PyDict_GetItemString(parameters, "regularize_scales");
		if(regularize_scales)
			if(PyFloat_Check(regularize_scales))
				params->regularizeScales = PyFloat_AsDouble(regularize_scales);
			else if(PyInt_Check(regularize_scales))
				params->regularizeScales = static_cast<double>(PyFloat_AsDouble(regularize_scales));
			else
				throw Exception("regularize_scales should be of type `float`.");

		PyObject* regularize_mean = PyDict_GetItemString(parameters, "regularize_mean");
		if(regularize_mean)
			if(PyFloat_Check(regularize_mean))
				params->regularizeMean = PyFloat_AsDouble(regularize_mean);
			else if(PyInt_Check(regularize_mean))
				params->regularizeMean = static_cast<double>(PyFloat_AsDouble(regularize_mean));
			else
				throw Exception("regularize_mean should be of type `float`.");
	}

	return params;
}



Mixture::Parameters* PyObject_ToMixtureParameters(PyObject* parameters) {
	Mixture::Parameters* params = new Mixture::Parameters;

	if(parameters && parameters != Py_None) {
		PyObject* verbosity = PyDict_GetItemString(parameters, "verbosity");
		if(verbosity)
			if(PyInt_Check(verbosity))
				params->verbosity = PyInt_AsLong(verbosity);
			else if(PyFloat_Check(verbosity))
				params->verbosity = static_cast<int>(PyFloat_AsDouble(verbosity));
			else
				throw Exception("verbosity should be of type `int`.");

		PyObject* max_iter = PyDict_GetItemString(parameters, "max_iter");
		if(max_iter)
			if(PyInt_Check(max_iter))
				params->maxIter = PyInt_AsLong(max_iter);
			else if(PyFloat_Check(max_iter))
				params->maxIter = static_cast<int>(PyFloat_AsDouble(max_iter));
			else
				throw Exception("max_iter should be of type `int`.");

		PyObject* threshold = PyDict_GetItemString(parameters, "threshold");
		if(threshold)
			if(PyFloat_Check(threshold))
				params->threshold = PyFloat_AsDouble(threshold);
			else if(PyInt_Check(threshold))
				params->threshold = static_cast<double>(PyFloat_AsDouble(threshold));
			else
				throw Exception("threshold should be of type `float`.");

		PyObject* val_iter = PyDict_GetItemString(parameters, "val_iter");
		if(val_iter)
			if(PyInt_Check(val_iter))
				params->valIter = PyInt_AsLong(val_iter);
			else if(PyFloat_Check(val_iter))
				params->valIter = static_cast<int>(PyFloat_AsDouble(val_iter));
			else
				throw Exception("val_iter should be of type `int`.");

		PyObject* val_look_ahead = PyDict_GetItemString(parameters, "val_look_ahead");
		if(val_look_ahead)
			if(PyInt_Check(val_look_ahead))
				params->valLookAhead = PyInt_AsLong(val_look_ahead);
			else if(PyFloat_Check(val_look_ahead))
				params->valLookAhead = static_cast<int>(PyFloat_AsDouble(val_look_ahead));
			else
				throw Exception("val_look_ahead should be of type `int`.");

		PyObject* initialize = PyDict_GetItemString(parameters, "initialize");
		if(initialize)
			if(PyBool_Check(initialize))
				params->initialize = (initialize == Py_True);
			else
				throw Exception("initialize should be of type `bool`.");

		PyObject* train_priors = PyDict_GetItemString(parameters, "train_priors");
		if(train_priors)
			if(PyBool_Check(train_priors))
				params->trainPriors = (train_priors == Py_True);
			else
				throw Exception("train_priors should be of type `bool`.");

		PyObject* train_components = PyDict_GetItemString(parameters, "train_components");
		if(train_components)
			if(PyBool_Check(train_components))
				params->trainComponents = (train_components == Py_True);
			else
				throw Exception("train_components should be of type `bool`.");

		PyObject* regularize_priors = PyDict_GetItemString(parameters, "regularize_priors");
		if(regularize_priors)
			if(PyFloat_Check(regularize_priors))
				params->regularizePriors = PyFloat_AsDouble(regularize_priors);
			else if(PyInt_Check(regularize_priors))
				params->regularizePriors = static_cast<double>(PyFloat_AsDouble(regularize_priors));
			else
				throw Exception("regularize_priors should be of type `float`.");
	}

	return params;
}



const char* Mixture_train_doc =
	"train(self, data, data_valid=None, parameters=None, component_parameters=None)\n"
	"\n"
	"Fits the parameters of the mixture distribution to the given data using expectation maximization (EM).\n"
	"\n"
	"The following example demonstrates possible parameters and default settings.\n"
	"\n"
	"\t>>> model.train(data, data_valid, \n"
	"\t>>> \tparameters={\n"
	"\t>>> \t\t'verbosity': 1,\n"
	"\t>>> \t\t'max_iter': 20,\n"
	"\t>>> \t\t'threshold': 1e-8,\n"
	"\t>>> \t\t'val_iter': 2,\n"
	"\t>>> \t\t'val_look_ahead': 5,\n"
	"\t>>> \t\t'initialize': True,\n"
	"\t>>> \t\t'train_priors': True,\n"
	"\t>>> \t\t'train_components': True,\n"
	"\t>>> \t\t'regularize_priors': 0.,\n"
	"\t>>> \t},\n"
	"\t>>> \tcomponent_parameters={\n"
	"\t>>> \t\t'verbosity': 0,\n"
	"\t>>> \t\t'max_iter': 10,\n"
	"\t>>> \t\t'train_priors': True,\n"
	"\t>>> \t\t'train_mean': True,\n"
	"\t>>> \t\t'train_covariance': True,\n"
	"\t>>> \t\t'train_scales': True,\n"
	"\t>>> \t\t'regularize_priors': 0.,\n"
	"\t>>> \t\t'regularize_mean': 0.,\n"
	"\t>>> \t\t'regularize_covariance': 0.,\n"
	"\t>>> \t\t'regularize_scales': 0.,\n"
	"\t>>> \t},\n"
	"\t>>> })\n"
	"\n"
	"@type  data: C{ndarray}\n"
	"@param data: data points stored in columns\n"
	"\n"
	"@type  data_valid: C{ndarray}\n"
	"@param data_valid: validation data used for early stopping\n"
	"\n"
	"@type  parameters: C{dict}\n"
	"@param parameters: hyperparameters controlling optimization and regularization\n"
	"\n"
	"@type  component_parameters: C{dict}\n"
	"@param component_parameters: hyperparameters passed down to components during M-step\n"
	"\n"
	"@rtype: C{bool}\n"
	"@return: C{True} if training converged, otherwise C{False}";

PyObject* Mixture_train(MixtureObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"data", "data_valid", "parameters", "component_parameters", 0};

	PyObject* data;
	PyObject* data_valid = 0;
	PyObject* parameters = 0;
	PyObject* component_parameters = 0;

	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O|OOO", const_cast<char**>(kwlist),
		&data, &data_valid, &parameters, &component_parameters))
		return 0;

	if(data_valid == Py_None)
		data_valid = 0;

	data = PyArray_FROM_OTF(data, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
	
	if(!data) {
		PyErr_SetString(PyExc_TypeError, "Data should be stored in a Numpy array.");
		return 0;
	}

	if(data_valid) {
		data_valid = PyArray_FROM_OTF(data_valid, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
		
		if(!data_valid) {
			PyErr_SetString(PyExc_TypeError, "Validation data should be stored in a Numpy array.");
			Py_DECREF(data);
			return 0;
		}
	}

	bool converged;

	try {
		Mixture::Parameters* params = PyObject_ToMixtureParameters(parameters);
		Mixture::Component::Parameters* component_params =
			PyObject_ToMixtureComponentParameters(component_parameters);

		if(data_valid)
			converged = self->mixture->train(
				PyArray_ToMatrixXd(data),
				PyArray_ToMatrixXd(data_valid),
				*params,
				*component_params);
		else
			converged = self->mixture->train(PyArray_ToMatrixXd(data), *params, *component_params);

		delete params;
		delete component_params;
	} catch(Exception exception) {
		Py_DECREF(data);
		Py_XDECREF(data_valid);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	Py_DECREF(data);
	Py_XDECREF(data_valid);

	if(converged) {
		Py_INCREF(Py_True);
		return Py_True;
	} else {
		Py_INCREF(Py_False);
		return Py_False;
	}
}



const char* Mixture_initialize_doc =
	"train(self, data, parameters=None, component_parameters=None)\n"
	"\n"
	"Calls L{initialize()} on all mixture components and resets all prior weights.\n"
	"\n"
	"@type  data: C{ndarray}\n"
	"@param data: data points stored in columns\n"
	"\n"
	"@type  parameters: C{dict}\n"
	"@param parameters: hyperparameters controlling optimization and regularization\n"
	"\n"
	"@type  component_parameters: C{dict}\n"
	"@param component_parameters: hyperparameters passed down to mixture components";

PyObject* Mixture_initialize(MixtureObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"data", "parameters", "component_parameters", 0};

	PyObject* data;
	PyObject* parameters = 0;
	PyObject* component_parameters = 0;

	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O|OO", const_cast<char**>(kwlist),
		&data, &parameters, &component_parameters))
		return 0;

	data = PyArray_FROM_OTF(data, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
	
	if(!data) {
		PyErr_SetString(PyExc_TypeError, "Data should be stored in a Numpy array.");
		return 0;
	}

	try {
		Mixture::Parameters* params = PyObject_ToMixtureParameters(parameters);
		Mixture::Component::Parameters* component_params =
			PyObject_ToMixtureComponentParameters(component_parameters);
		self->mixture->initialize(PyArray_ToMatrixXd(data), *params, *component_params);
		delete params;
		delete component_params;
	} catch(Exception exception) {
		Py_DECREF(data);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	Py_DECREF(data);

	Py_INCREF(Py_None);
	return Py_None;
}


PyObject* Mixture_subscript(MixtureObject* self, PyObject* key) {
	if(!PyInt_Check(key)) {
		PyErr_SetString(PyExc_TypeError, "Index must be an integer.");
		return 0;
	}

	int i = PyInt_AsLong(key);

	PyObject* component;

	if(i < self->componentTypes.size())
		component = Distribution_new(self->componentTypes[i], 0, 0);
	else
		component = Distribution_new(&MixtureComponent_type, 0, 0);

	try {
		reinterpret_cast<MixtureComponentObject*>(component)->component =
			self->mixture->operator[](PyInt_AsLong(key));
		reinterpret_cast<MixtureComponentObject*>(component)->owner = false;
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	Py_INCREF(component);
	return component;
}



PyObject* Mixture_add_component(MixtureObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"component", 0};

	PyObject* component;

	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O!", const_cast<char**>(kwlist),
		&MixtureComponent_type, &component))
		return 0;

	try {
		self->componentTypes.push_back(Py_TYPE(component));
		self->mixture->addComponent(
			reinterpret_cast<MixtureComponentObject*>(component)->component->copy());
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	Py_INCREF(Py_None);
	return Py_None;
}



PyObject* Mixture_priors(MixtureObject* self, void*) {
	PyObject* array = PyArray_FromMatrixXd(self->mixture->priors());

	// make array immutable
	reinterpret_cast<PyArrayObject*>(array)->flags &= ~NPY_WRITEABLE;

	return array;
}



int Mixture_set_priors(MixtureObject* self, PyObject* value, void*) {
	value = PyArray_FROM_OTF(value, NPY_DOUBLE, NPY_IN_ARRAY);

	if(!value) {
		PyErr_SetString(PyExc_TypeError, "Scales should be of type `ndarray`.");
		return -1;
	}

	try {
		self->mixture->setPriors(PyArray_ToMatrixXd(value));
	} catch(Exception exception) {
		Py_DECREF(value);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	Py_DECREF(value);

	return 0;
}



PyObject* MoGSM_reduce(MoGSMObject* self, PyObject*) {
	// constructor arguments
	PyObject* args = Py_BuildValue("(iii)",
		self->mixture->dim(), 
		0,
		self->mixture->numScales());

	PyObject* components = PyTuple_New(self->mixture->numComponents());

	for(int k = 0; k < self->mixture->numComponents(); ++k) {
		PyObject* component = Mixture_subscript(
			reinterpret_cast<MixtureObject*>(self), PyInt_FromLong(k));
		PyTuple_SetItem(components, k, component);
	}

	PyObject* state = Py_BuildValue("(Ob)", components, self->mixture->initialized());
	PyObject* result = Py_BuildValue("(OOO)", Py_TYPE(self), args, state);

	Py_DECREF(components);
	Py_DECREF(args);
	Py_DECREF(state);

	return result;
}



PyObject* Mixture_reduce(MixtureObject* self, PyObject*) {
	// constructor arguments
	PyObject* args = Py_BuildValue("(i)", self->mixture->dim());

	PyObject* components = PyTuple_New(self->mixture->numComponents());

	for(int k = 0; k < self->mixture->numComponents(); ++k) {
		PyObject* component = Mixture_subscript(self, PyInt_FromLong(k));
		PyTuple_SetItem(components, k, component);
	}

	PyObject* state = Py_BuildValue("(Ob)", components, self->mixture->initialized());
	PyObject* result = Py_BuildValue("(OOO)", Py_TYPE(self), args, state);

	Py_DECREF(components);
	Py_DECREF(args);
	Py_DECREF(state);

	return result;
}



PyObject* Mixture_setstate(MixtureObject* self, PyObject* state) {
	PyObject* components;
	bool initialized;

	PyArg_ParseTuple(state, "(Ob)", &components, &initialized);

	self->mixture->setInitialized(initialized);

	for(int k = 0; k < PyTuple_Size(components); ++k) {
		PyObject* component = PyTuple_GetItem(components, k);
		PyObject* args = Py_BuildValue("(O)", component);
		PyObject* kwds = Py_BuildValue("{}");

		Mixture_add_component(self, args, kwds);

		Py_DECREF(args);
		Py_DECREF(kwds);
	}

	Py_INCREF(Py_None);
	return Py_None;
}



const char* MixtureComponent_doc =
	"Abstract base class for objects which can be used as mixture components.";

int MixtureComponent_init(MixtureComponentObject*, PyObject*, PyObject*) {
	PyErr_SetString(PyExc_NotImplementedError, "This is an abstract class.");
	return -1;
}



const char* MixtureComponent_train_doc =
	"train(self, data, weights=None, parameters=None)\n"
	"\n"
	"Trains the parameters of the distribution, possibly weighting data points.\n"
	"\n"
	"@type  data: C{ndarray}\n"
	"@param data: data points stored in columns\n"
	"\n"
	"@type  weights: C{ndarray}\n"
	"@param weights: weights which sum to one\n"
	"\n"
	"@type  parameters: C{dict}\n"
	"@param parameters: hyperparameters controlling optimization and regularization\n"
	"\n"
	"@rtype: C{bool}\n"
	"@return: C{True} if training converged, otherwise C{False}";

PyObject* MixtureComponent_train(MixtureComponentObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"data", "weights", "parameters", 0};

	PyObject* data;
	PyObject* weights = 0;
	PyObject* parameters = 0;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O|OO", const_cast<char**>(kwlist),
		&data, &weights, &parameters))
		return 0;

	if(weights == Py_None)
		weights = 0;

	// make sure data is stored in NumPy array
	data = PyArray_FROM_OTF(data, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

	if(!data) {
		PyErr_SetString(PyExc_TypeError, "Data has to be stored in NumPy array.");
		return 0;
	}

	bool converged;

	try {
		Mixture::Component::Parameters* params = PyObject_ToMixtureComponentParameters(parameters);

		if(weights) {
			MatrixXd weightsMatrix = PyArray_ToMatrixXd(weights);
			MatrixXd dataMatrix = PyArray_ToMatrixXd(data);

			if(weightsMatrix.rows() != 1)
				weightsMatrix = weightsMatrix.transpose();
			if(weightsMatrix.rows() != 1)
				throw Exception("Weights should be stored in a row vector.");

			converged = self->component->train(dataMatrix, weightsMatrix, *params);
		} else {
			converged = self->component->train(PyArray_ToMatrixXd(data), *params);
		}

		delete params;
	} catch(Exception exception) {
		Py_DECREF(data);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	Py_DECREF(data);

	if(converged) {
		Py_INCREF(Py_True);
		return Py_True;
	} else {
		Py_INCREF(Py_False);
		return Py_False;
	}
}



const char* MixtureComponent_initialize_doc =
	"train(self, data, parameters=None)\n"
	"\n"
	"Tries to guess reasonable parameters based on the data.\n"
	"\n"
	"@type  data: C{ndarray}\n"
	"@param data: data points stored in columns\n"
	"\n"
	"@type  parameters: C{dict}\n"
	"@param parameters: hyperparameters controlling optimization and regularization";

PyObject* MixtureComponent_initialize(MixtureComponentObject* self, PyObject* args, PyObject* kwds) {
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
		PyErr_SetString(PyExc_TypeError, "Data has to be stored in NumPy array.");
		return 0;
	}

	try {
		Mixture::Component::Parameters* params = PyObject_ToMixtureComponentParameters(parameters);
		self->component->initialize(PyArray_ToMatrixXd(data), *params);
		delete params;
	} catch(Exception exception) {
		Py_DECREF(data);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	Py_DECREF(data);

	Py_INCREF(Py_None);
	return Py_None;
}
