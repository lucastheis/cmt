#include "trainableinterface.h"

#include "cmt/utils"
using CMT::Exception;

#include "Eigen/Core"
using Eigen::Dynamic;
using Eigen::Matrix;
using Eigen::Map;

#include <iostream>
using std::cout;
using std::endl;

Trainable::Parameters* PyObject_ToParameters(PyObject* parameters) {
	return PyObject_ToParameters(parameters, new Trainable::Parameters);
}



Trainable::Parameters* PyObject_ToParameters(
	PyObject* parameters,
	Trainable::Parameters* params)
{
	// read parameters from dictionary
	if(parameters && parameters != Py_None) {
		if(!PyDict_Check(parameters))
			throw Exception("Parameters should be stored in a dictionary.");

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
				params->threshold = static_cast<double>(PyInt_AsLong(threshold));
			else
				throw Exception("threshold should be of type `float`.");

		PyObject* num_grad = PyDict_GetItemString(parameters, "num_grad");
		if(num_grad)
			if(PyInt_Check(num_grad))
				params->numGrad = PyInt_AsLong(num_grad);
			else if(PyFloat_Check(num_grad))
				params->numGrad = static_cast<int>(PyFloat_AsDouble(num_grad));
			else
				throw Exception("num_grad should be of type `int`.");

		PyObject* batch_size = PyDict_GetItemString(parameters, "batch_size");
		if(batch_size)
			if(PyInt_Check(batch_size))
				params->batchSize = PyInt_AsLong(batch_size);
			else if(PyFloat_Check(batch_size))
				params->batchSize = static_cast<int>(PyFloat_AsDouble(batch_size));
			else
				throw Exception("batch_size should be of type `int`.");

		PyObject* cb_iter = PyDict_GetItemString(parameters, "cb_iter");
		if(cb_iter)
			if(PyInt_Check(cb_iter))
				params->cbIter = PyInt_AsLong(cb_iter);
			else if(PyFloat_Check(cb_iter))
				params->cbIter = static_cast<int>(PyFloat_AsDouble(cb_iter));
			else
				throw Exception("cb_iter should be of type `int`.");

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
	}

	return params;
}



const char* Trainable_initialize_doc =
	"initialize(self, input, output)\n"
	"\n"
	"Tries to guess more sensible initial values for the model parameters from data.\n"
	"\n"
	"@type  input: C{ndarray}\n"
	"@param input: inputs stored in columns\n"
	"\n"
	"@type  output: C{ndarray}\n"
	"@param output: outputs stored in columns";

PyObject* Trainable_initialize(TrainableObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"input", "output", 0};

	PyObject* input;
	PyObject* output;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "OO", const_cast<char**>(kwlist),
		&input, &output))
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
		self->distribution->initialize(
			PyArray_ToMatrixXd(input), 
			PyArray_ToMatrixXd(output));
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



PyObject* Trainable_train(
	TrainableObject* self,
	PyObject* args,
	PyObject* kwds,
	Trainable::Parameters* (*PyObject_ToParameters)(PyObject*))
{
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

	if((input_val && PyDict_Check(input_val)) || (output_val && PyDict_Check(output_val))) {
		// for some reason PyArray_FROM_OTF segfaults when input_val is a dictionary
		Py_DECREF(input);
		Py_DECREF(output);
		PyErr_SetString(PyExc_TypeError, "Validation data has to be stored in NumPy arrays.");
		return 0;
	}

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

		Trainable::Parameters* params = PyObject_ToParameters(parameters);

		if(input_val && output_val) {
			converged = self->distribution->train(
				PyArray_ToMatrixXd(input), 
				PyArray_ToMatrixXd(output), 
				PyArray_ToMatrixXd(input_val),
				PyArray_ToMatrixXd(output_val),
				*params);
		} else {
			converged = self->distribution->train(
				PyArray_ToMatrixXd(input), 
				PyArray_ToMatrixXd(output), 
				*params);
		}

		delete params;

		Py_DECREF(input);
		Py_DECREF(output);
		Py_XDECREF(input_val);
		Py_XDECREF(output_val);

		if(converged) {
			Py_INCREF(Py_True);
			return Py_True;
		} else {
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



const char* Trainable_parameters_doc =
	"parameters(self, parameters=None)\n"
	"\n"
	"Summarizes the parameters of the model in a long vector.\n"
	"\n"
	"If C{parameters} is given, only the parameters with C{train_* = True} will be contained "
	"in the vector.\n"
	"\n"
	"@type  parameters: C{dict}\n"
	"@param parameters: a dictionary containing hyperparameters\n"
	"\n"
	"@rtype: C{ndarray}\n"
	"@return: model parameters vectorized and concatenated";

PyObject* Trainable_parameters(
	TrainableObject* self,
	PyObject* args,
	PyObject* kwds,
	Trainable::Parameters* (*PyObject_ToParameters)(PyObject*))
{
	const char* kwlist[] = {"parameters", 0};

	PyObject* parameters = 0;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "|O", const_cast<char**>(kwlist), &parameters))
		return 0;

	try {
		Trainable::Parameters* params = PyObject_ToParameters(parameters);

		lbfgsfloatval_t* x = self->distribution->parameters(*params);

		PyObject* xObj = PyArray_FromMatrixXd(
			Map<Matrix<lbfgsfloatval_t, Dynamic, Dynamic> >(
				x, self->distribution->numParameters(*params), 1));

		lbfgs_free(x);

		return xObj;
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	return 0;
}



const char* Trainable_set_parameters_doc =
	"_set_parameters(self, x, parameters=None)\n"
	"\n"
	"Loads all model parameters from a vector as produced by L{parameters()}.\n"
	"\n"
	"@type  x: C{ndarray}\n"
	"@param x: all model parameters concatenated to a vector\n"
	"\n"
	"@type  parameters: C{dict}\n"
	"@param parameters: a dictionary containing hyperparameters";

PyObject* Trainable_set_parameters(
	TrainableObject* self,
	PyObject* args,
	PyObject* kwds,
	Trainable::Parameters* (*PyObject_ToParameters)(PyObject*))
{
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
		Trainable::Parameters* params = PyObject_ToParameters(parameters);

		self->distribution->setParameters(reinterpret_cast<double*>(PyArray_DATA(x)), *params);

		delete params;

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



const char* Trainable_check_gradient_doc =
	"_check_gradient(self, input, output, epsilon=1e-5, parameters=None)\n"
	"\n"
	"Compare the gradient to a numerical gradient.\n"
	"\n"
	"Numerically estimate the gradient using finite differences and return the\n"
	"norm of the difference between the numerical gradient and the gradient\n"
	"used during training. This method is used for testing purposes.\n"
	"\n"
	"@type  input: C{ndarray}\n"
	"@param input: inputs stored in columns\n"
	"\n"
	"@type  output: C{ndarray}\n"
	"@param output: inputs stored in columns\n"
	"\n"
	"@type  epsilon: C{float}\n"
	"@param epsilon: a small change added to the current parameters\n"
	"\n"
	"@type  parameters: C{dict}\n"
	"@param parameters: a dictionary containing hyperparameters\n"
	"\n"
	"@rtype: C{float}\n"
	"@return: difference between numerical and analytical gradient";

PyObject* Trainable_check_gradient(
	TrainableObject* self,
	PyObject* args,
	PyObject* kwds,
	Trainable::Parameters* (*PyObject_ToParameters)(PyObject*))
{
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
		Trainable::Parameters* params = PyObject_ToParameters(parameters);

		double err = self->distribution->checkGradient(
			PyArray_ToMatrixXd(input),
			PyArray_ToMatrixXd(output),
			epsilon,
			*params);

		delete params;

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



const char* Trainable_parameter_gradient_doc =
	"_parameter_gradient(self, input, output, x=None, parameters=None)\n"
	"\n"
	"Computes the gradient of the parameters as returned by L{_parameters()}.\n"
	"\n"
	"If C{x} is not specified, the gradient will be evaluated for the current\n"
	"parameters of the model.\n"
	"\n"
	"@type  input: C{ndarray}\n"
	"@param input: inputs stored in columns\n"
	"\n"
	"@type  output: C{ndarray}\n"
	"@param output: outputs stored in columns\n"
	"\n"
	"@type  x: C{ndarray}\n"
	"@param x: set of parameters for which to evaluate gradient\n"
	"\n"
	"@type  parameters: C{dict}\n"
	"@param parameters: a dictionary containing hyperparameters\n"
	"\n"
	"@rtype: C{ndarray}\n"
	"@return: gradient of model parameters";

PyObject* Trainable_parameter_gradient(
	TrainableObject* self,
	PyObject* args,
	PyObject* kwds,
	Trainable::Parameters* (*PyObject_ToParameters)(PyObject*))
{
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

	if(x == Py_None)
		x = 0;

	if(x)
		x = PyArray_FROM_OTF(x, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

	// for performance reasons, only perform these checks in the interface
	if(PyArray_DIM(input, 0) != self->distribution->dimIn()) {
		PyErr_SetString(PyExc_RuntimeError, "Input has wrong dimensionality.");
		return 0;
	}

	if(PyArray_DIM(output, 0) != self->distribution->dimOut()) {
		PyErr_SetString(PyExc_RuntimeError, "Output has wrong dimensionality.");
		return 0;
	}

	if(PyArray_DIM(output, 1) != PyArray_DIM(input, 1)) {
		PyErr_SetString(PyExc_RuntimeError, "Number of inputs and outputs should be the same.");
		return 0;
	}

	try {
		Trainable::Parameters* params = PyObject_ToParameters(parameters);

		MatrixXd gradient(self->distribution->numParameters(*params), 1);

		if(x) {
			#if LBFGS_FLOAT == 64
			self->distribution->parameterGradient(
				PyArray_ToMatrixXd(input),
				PyArray_ToMatrixXd(output),
				reinterpret_cast<lbfgsfloatval_t*>(PyArray_DATA(x)),
				gradient.data(),
				*params);
			#elif LBFGS_FLOAT == 32
			lbfgsfloatval_t* xLBFGS = lbfgs_malloc(PyArray_SIZE(x));
			lbfgsfloatval_t* gLBFGS = lbfgs_malloc(PyArray_SIZE(x));
			double* xData = reinterpret_cast<double*>(PyArray_DATA(x));

			for(int i = 0; i < PyArray_SIZE(x); ++i)
				xLBFGS[i] = static_cast<lbfgsfloatval_t>(xData[i]);

			self->distribution->parameterGradient(
				PyArray_ToMatrixXd(input),
				PyArray_ToMatrixXd(output),
				xLBFGS,
				gLBFGS,
				*params);

			for(int i = 0; i < PyArray_SIZE(x); ++i)
				gradient.data()[i] = static_cast<double>(gLBFGS[i]);

			lbfgs_free(gLBFGS);
			lbfgs_free(xLBFGS);
			#else
			#error "LibLBFGS is configured in a way I don't understand."
			#endif
		} else {
			lbfgsfloatval_t* x = self->distribution->parameters(*params);

			self->distribution->parameterGradient(
				PyArray_ToMatrixXd(input),
				PyArray_ToMatrixXd(output),
				x,
				gradient.data(),
				*params);

			lbfgs_free(x);
		}

		delete params;

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



const char* Trainable_fisher_information_doc =
	"_parameter_gradient(self, input, output, x=None, parameters=None)\n"
	"\n"
	"Estimates the Fisher information matrix of the parameters as returned by L{_parameters()}.\n"
	"\n"
	"@type  input: C{ndarray}\n"
	"@param input: inputs stored in columns\n"
	"\n"
	"@type  output: C{ndarray}\n"
	"@param output: outputs stored in columns\n"
	"\n"
	"@type  parameters: C{dict}\n"
	"@param parameters: a dictionary containing hyperparameters\n"
	"\n"
	"@rtype: C{ndarray}\n"
	"@return: the Fisher information matrix";

PyObject* Trainable_fisher_information(
	TrainableObject* self,
	PyObject* args,
	PyObject* kwds,
	Trainable::Parameters* (*PyObject_ToParameters)(PyObject*))
{
	const char* kwlist[] = {"input", "output", "parameters", 0};

	PyObject* input;
	PyObject* output;
	PyObject* parameters = 0;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "OO|O", const_cast<char**>(kwlist),
		&input, &output, &parameters))
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
		Trainable::Parameters* params = PyObject_ToParameters(parameters);

		MatrixXd fisherInformation = self->distribution->fisherInformation(
			PyArray_ToMatrixXd(input),
			PyArray_ToMatrixXd(output),
			*params);

		delete params;

		Py_DECREF(input);
		Py_DECREF(output);

		return PyArray_FromMatrixXd(fisherInformation);
	} catch(Exception exception) {
		Py_DECREF(input);
		Py_DECREF(output);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}
}



const char* Trainable_check_performance_doc =
	"_check_performance(self, input, output, repetitions=2, parameters=None)\n"
	"\n"
	"Measures the time it takes to evaluate the parameter gradient for the given data points.\n"
	"\n"
	"This function can be used to tune the C{batch_size} parameter.\n"
	"\n"
	"@type  input: C{ndarray}\n"
	"@param input: inputs stored in columns\n"
	"\n"
	"@type  output: C{ndarray}\n"
	"@param output: outputs stored in columns\n"
	"\n"
	"@type  repetitions: C{int}\n"
	"@param repetitions: number of times the gradient is evaluated before averaging\n"
	"\n"
	"@type  parameters: C{dict}\n"
	"@param parameters: a dictionary containing hyperparameters\n"
	"\n"
	"@rtype: C{ndarray}\n"
	"@return: estimated time for one gradient evaluation";

PyObject* Trainable_check_performance(
	TrainableObject* self,
	PyObject* args,
	PyObject* kwds,
	Trainable::Parameters* (*PyObject_ToParameters)(PyObject*))
{
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
		Trainable::Parameters* params = PyObject_ToParameters(parameters);

		double err = self->distribution->checkPerformance(
			PyArray_ToMatrixXd(input),
			PyArray_ToMatrixXd(output),
			repetitions, 
			*params);

		delete params;

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
