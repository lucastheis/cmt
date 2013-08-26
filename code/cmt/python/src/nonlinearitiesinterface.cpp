#include "pyutils.h"
#include "nonlinearitiesinterface.h"

#include "cmt/utils"
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
	"Abstract base class for nonlinear functions used, for example, by L{GLM<models.GLM>}.";

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



const char* Nonlinearity_reduce_doc =
	"__reduce__(self)\n"
	"\n"
	"Method used by Pickle.";

PyObject* Nonlinearity_reduce(NonlinearityObject* self, PyObject*) {
	PyObject* args = Py_BuildValue("()");
	PyObject* result = Py_BuildValue("(OO)", Py_TYPE(self), args);

	Py_DECREF(args);

	return result;
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



const char* ExponentialFunction_doc =
	"The exponential function.\n"
	"\n"
	"$$f(x) = e^{x}$$";

int ExponentialFunction_init(ExponentialFunctionObject* self, PyObject*, PyObject*) {
	try {
		self->nonlinearity = new ExponentialFunction;
		return 0;
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	return -1;
}



const char* HistogramNonlinearity_doc =
	"Histogram nonlinearity with $N$ bins.\n"
	"\n"
	"$$f(x) = \\varepsilon + \\begin{cases} h_0 & \\text{ if } x < b_0 \\\\ h_n & \\text{ if } b_n \\leq x < b_{n + 1} \\\\ h_N & \\text{ if } b_N \\leq x \\end{cases}$$\n"
	"\n"
	"@type  inputs: C{ndarray}\n"
	"@param inputs: example inputs to the nonlinearity\n"
	"\n"
	"@type  outputs: C{ndarray}\n"
	"@param outputs: example outputs to the nonlinearity\n"
	"\n"
	"@type  num_bins: C{int}\n"
	"@param num_bins: number of bins used to bin the data\n"
	"\n"
	"@type  bin_edges: C{list}\n"
	"@param bin_edges: ascending list of $N + 1$ bin edges\n"
	"\n"
	"@type  epsilon: C{float}\n"
	"@param epsilon: small offset for numerical stability, $\\varepsilon$";

int HistogramNonlinearity_init(HistogramNonlinearityObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {
		"inputs",
		"outputs",
		"num_bins",
		"bin_edges",
		"epsilon"
	};

	PyObject* inputs = 0;
	PyObject* outputs = 0;
	int num_bins = 0;
	PyObject* bin_edges = 0;
	double epsilon = 1e-12;

	if(!PyArg_ParseTupleAndKeywords(args, kwds, "|OOiOd", const_cast<char**>(kwlist),
		&inputs, &outputs, &num_bins, &bin_edges, &epsilon))
		return -1;

	if(num_bins <= 0 && bin_edges == 0) {
		PyErr_SetString(PyExc_TypeError, "Please specify either `num_bins` or `bin_edges`.");
		return -1;
	}

	if(inputs || outputs) {
		inputs = PyArray_FROM_OTF(inputs, NPY_DOUBLE, NPY_IN_ARRAY);
		outputs = PyArray_FROM_OTF(outputs, NPY_DOUBLE, NPY_IN_ARRAY);

		if(!inputs || !outputs) {
			PyErr_SetString(PyExc_TypeError, "Inputs and outputs should be of type `ndarray`.");
			return -1;
		}
	}

	if(bin_edges && !PySequence_Check(bin_edges)) {
		PyErr_SetString(PyExc_TypeError, "`bin_edges` should be of type `list` or similar.");
		return -1;
	}

	if(!bin_edges && (!inputs || !outputs)) {
		PyErr_SetString(PyExc_TypeError, "Please specify `bin_edges`.");
		return -1;
	}

	vector<double> binEdges;

	if(bin_edges) {
		// convert Python list/tuple/array into C++ vector
		binEdges = vector<double>(PySequence_Size(bin_edges));

		for(int i = 0; i < PySequence_Size(bin_edges); ++i) {
			PyObject* entry = PySequence_GetItem(bin_edges, i);
			double value = PyFloat_AsDouble(entry);

			if(PyErr_Occurred())
				return -1;

			binEdges.push_back(value);
		}
	}

	try {
		if(inputs && outputs) {
			if(bin_edges)
				self->nonlinearity = new HistogramNonlinearity(
					PyArray_ToMatrixXd(inputs),
					PyArray_ToMatrixXd(outputs),
					binEdges,
					epsilon);
			else
				self->nonlinearity = new HistogramNonlinearity(
					PyArray_ToMatrixXd(inputs),
					PyArray_ToMatrixXd(outputs),
					num_bins,
					epsilon);
		} else {
			self->nonlinearity = new HistogramNonlinearity(
				binEdges,
				epsilon);
		}

		return 0;
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	return -1;
}
