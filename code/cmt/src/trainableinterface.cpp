#include "exception.h"
#include "trainableinterface.h"

#include "Eigen/Core"
using Eigen::Dynamic;
using Eigen::Matrix;
using Eigen::Map;

#include <iostream>

Trainable::Parameters* PyObject_ToParameters(PyObject* parameters) {
	Trainable::Parameters* params = new Trainable::Parameters;

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
				params->threshold = static_cast<double>(PyFloat_AsDouble(threshold));
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
	}

	return params;
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
	"@type  parameters: dict\n"
	"@param parameters: a dictionary containing hyperparameters\n"
	"\n"
	"@rtype: ndarray\n"
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
	"set_parameters(self, x, parameters=None)\n"
	"\n"
	"Loads all model parameters from a vector as produced by L{parameters()}.\n"
	"\n"
	"@type  x: ndarray\n"
	"@param x: all model parameters concatenated to a vector\n"
	"\n"
	"@type  parameters: dict\n"
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

	if(x)
		x = PyArray_FROM_OTF(x, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

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
