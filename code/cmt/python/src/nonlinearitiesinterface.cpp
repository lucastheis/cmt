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

int LogisticFunction_init(LogisticFunctionObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"epsilon", 0};

	double epsilon = 1e-12;

	if(!PyArg_ParseTupleAndKeywords(args, kwds, "|d", const_cast<char**>(kwlist), &epsilon))
		return -1;

	try {
		self->nonlinearity = new LogisticFunction(epsilon);
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

int ExponentialFunction_init(ExponentialFunctionObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"epsilon", 0};

	double epsilon = 1e-12;

	if(!PyArg_ParseTupleAndKeywords(args, kwds, "|d", const_cast<char**>(kwlist), &epsilon))
		return -1;

	try {
		self->nonlinearity = new ExponentialFunction(epsilon);
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
		"epsilon",
		0
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



PyObject* HistogramNonlinearity_reduce(HistogramNonlinearityObject* self, PyObject*) {
	vector<double> binEdges = self->nonlinearity->binEdges();

	PyObject* bin_edges = PyList_New(binEdges.size());
	for(int i = 0; i < binEdges.size(); ++i)
		PyList_SetItem(bin_edges, i, PyFloat_FromDouble(binEdges[i]));

	PyObject* parameters = PyArray_FromMatrixXd(self->nonlinearity->parameters());

	// dummy inputs and outputs
	npy_intp dims[2] = {1, 1};
	PyObject* inputs = PyArray_Zeros(2, dims, PyArray_DescrFromType(NPY_DOUBLE), 1);
	PyObject* outputs = PyArray_Zeros(2, dims, PyArray_DescrFromType(NPY_DOUBLE), 1);

	double epsilon = self->nonlinearity->epsilon();

	PyObject* args = Py_BuildValue("(OOiOd)", inputs, outputs, 0, bin_edges, epsilon);
	PyObject* state = Py_BuildValue("(O)", parameters);
	PyObject* result = Py_BuildValue("(OOO)", Py_TYPE(self), args, state);

	Py_DECREF(args);
	Py_DECREF(state);

	return result;
}



PyObject* HistogramNonlinearity_setstate(HistogramNonlinearityObject* self, PyObject* state) {
	PyObject* parameters;

	if(!PyArg_ParseTuple(state, "(O)", &parameters))
		return 0;

	try {
		self->nonlinearity->setParameters(PyArray_ToMatrixXd(parameters));
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	Py_INCREF(Py_None);
	return Py_None;
}



const char* BlobNonlinearity_doc =
	"Mixture of Gaussian blobs.\n"
	"\n"
	"$$f(x) = \\varepsilon + \\sum_k \\alpha_k \\exp\\left(\\frac{\\lambda_k}{2} (x - \\mu_k)^2\\right)$$\n"
	"\n"
	"@type  num_components: C{int}\n"
	"@param num_components: number of Gaussian blobs\n"
	"\n"
	"@type  epsilon: C{float}\n"
	"@param epsilon: small offset for numerical stability, $\\varepsilon$";

int BlobNonlinearity_init(BlobNonlinearityObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"num_components", "epsilon", 0};

	int num_components = 3;
	double epsilon = 1e-12;

	if(!PyArg_ParseTupleAndKeywords(args, kwds, "|id", const_cast<char**>(kwlist),
		&num_components, &epsilon))
		return -1;

	try {
		self->nonlinearity = new BlobNonlinearity(num_components, epsilon);
		return 0;
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	return -1;
}



PyObject* BlobNonlinearity_reduce(BlobNonlinearityObject* self, PyObject*) {
	PyObject* parameters = PyArray_FromMatrixXd(self->nonlinearity->parameters());

	PyObject* args = Py_BuildValue("(id)", self->nonlinearity->numComponents(), self->nonlinearity->epsilon());
	PyObject* state = Py_BuildValue("(O)", parameters);
	PyObject* result = Py_BuildValue("(OOO)", Py_TYPE(self), args, state);

	Py_DECREF(args);
	Py_DECREF(state);

	return result;
}



PyObject* BlobNonlinearity_setstate(BlobNonlinearityObject* self, PyObject* state) {
	PyObject* parameters;

	if(!PyArg_ParseTuple(state, "(O)", &parameters))
		return 0;

	try {
		self->nonlinearity->setParameters(PyArray_ToMatrixXd(parameters));
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	Py_INCREF(Py_None);
	return Py_None;
}



const char* TanhBlobNonlinearity_doc =
	"Hyperbolic tangent applied to Gaussian blob nonlinearity.\n"
	"\n"
	"$$f(x) = \\tanh\\left(\\varepsilon + \\sum_k \\alpha_k \\exp\\left(\\frac{\\lambda_k}{2} (x - \\mu_k)^2\\right)\\right)$$\n"
	"\n"
	"@type  num_components: C{int}\n"
	"@param num_components: number of Gaussian blobs\n"
	"\n"
	"@type  epsilon: C{float}\n"
	"@param epsilon: small offset for numerical stability, $\\varepsilon$";

int TanhBlobNonlinearity_init(TanhBlobNonlinearityObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"num_components", "epsilon", 0};

	int num_components = 3;
	double epsilon = 1e-12;

	if(!PyArg_ParseTupleAndKeywords(args, kwds, "|id", const_cast<char**>(kwlist),
		&num_components, &epsilon))
		return -1;

	try {
		self->nonlinearity = new TanhBlobNonlinearity(num_components, epsilon);
		return 0;
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	return -1;
}



PyObject* TanhBlobNonlinearity_reduce(TanhBlobNonlinearityObject* self, PyObject*) {
	PyObject* parameters = PyArray_FromMatrixXd(self->nonlinearity->parameters());

	PyObject* args = Py_BuildValue("(id)", self->nonlinearity->numComponents(), self->nonlinearity->epsilon());
	PyObject* state = Py_BuildValue("(O)", parameters);
	PyObject* result = Py_BuildValue("(OOO)", Py_TYPE(self), args, state);

	Py_DECREF(args);
	Py_DECREF(state);

	return result;
}



PyObject* TanhBlobNonlinearity_setstate(TanhBlobNonlinearityObject* self, PyObject* state) {
	PyObject* parameters;

	if(!PyArg_ParseTuple(state, "(O)", &parameters))
		return 0;

	try {
		self->nonlinearity->setParameters(PyArray_ToMatrixXd(parameters));
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	Py_INCREF(Py_None);
	return Py_None;
}
