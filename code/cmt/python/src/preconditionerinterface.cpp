#include "preconditionerinterface.h"

#include <utility>
using std::pair;
using std::make_pair;

#include <new>
using std::bad_alloc;

#include "cmt/utils"
using CMT::Exception;

PyObject* Preconditioner_call(PreconditionerObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"input", "output", 0};

	PyObject* input;
	PyObject* output = 0;

	if(PyArg_ParseTupleAndKeywords(args, kwds, "O|O", const_cast<char**>(kwlist), &input, &output))
		if(output && output != Py_None) {
			input = PyArray_FROM_OTF(input, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
			output = PyArray_FROM_OTF(output, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

			if(!input || !output) {
				Py_XDECREF(input);
				Py_XDECREF(output);
				PyErr_SetString(PyExc_TypeError, "Input and output should be of type `ndarray`.");
				return 0;
			}

			try {
				pair<ArrayXXd, ArrayXXd> data = self->preconditioner->operator()(
					PyArray_ToMatrixXd(input),
					PyArray_ToMatrixXd(output));

				PyObject* inputObj = PyArray_FromMatrixXd(data.first);
				PyObject* outputObj = PyArray_FromMatrixXd(data.second);

				PyObject* tuple = Py_BuildValue("(OO)", inputObj, outputObj);

				Py_DECREF(input);
				Py_DECREF(output);
				Py_DECREF(inputObj);
				Py_DECREF(outputObj);

				return tuple;

			} catch(Exception exception) {
				Py_DECREF(input);
				Py_DECREF(output);
				PyErr_SetString(PyExc_RuntimeError, exception.message());
				return 0;
			}
		} else {
			input = PyArray_FROM_OTF(input, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

			if(!input) {
				Py_XDECREF(input);
				PyErr_SetString(PyExc_TypeError, "Input should be of type `ndarray`.");
				return 0;
			}

			try {
				PyObject* inputObj = PyArray_FromMatrixXd(
					self->preconditioner->operator()(PyArray_ToMatrixXd(input)));
				Py_DECREF(input);
				return inputObj;
			} catch(Exception exception) {
				Py_DECREF(input);
				PyErr_SetString(PyExc_RuntimeError, exception.message());
				return 0;
			}
		}

	return 0;
}



const char* Preconditioner_inverse_doc =
	"inverse(self, input, output=None)\n"
	"\n"
	"Computes original inputs and outputs from transformed inputs and outputs."
	"\n"
	"@type  input: C{ndarray}\n"
	"@param input: preconditioned inputs stored in columns\n"
	"\n"
	"@type  output: C{ndarray}\n"
	"@param output: preconditioned outputs stored in columns\n"
	"\n"
	"@rtype: tuple/C{ndarray}\n"
	"@return: tuple or array containing inputs or inputs and outputs, respectively";

PyObject* Preconditioner_inverse(PreconditionerObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"input", "output", 0};

	PyObject* input;
	PyObject* output = 0;

	if(PyArg_ParseTupleAndKeywords(args, kwds, "O|O", const_cast<char**>(kwlist), &input, &output))
		if(output && output != Py_None) {
			input = PyArray_FROM_OTF(input, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
			output = PyArray_FROM_OTF(output, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

			if(!input || !output) {
				Py_XDECREF(input);
				Py_XDECREF(output);
				PyErr_SetString(PyExc_TypeError, "Input and output should be of type `ndarray`.");
			}

			try {
				pair<ArrayXXd, ArrayXXd> data = self->preconditioner->inverse(
					PyArray_ToMatrixXd(input),
					PyArray_ToMatrixXd(output));

				PyObject* inputObj = PyArray_FromMatrixXd(data.first);
				PyObject* outputObj = PyArray_FromMatrixXd(data.second);

				PyObject* tuple = Py_BuildValue("(OO)", inputObj, outputObj);

				Py_DECREF(input);
				Py_DECREF(output);
				Py_DECREF(inputObj);
				Py_DECREF(outputObj);

				return tuple;

			} catch(Exception exception) {
				Py_DECREF(input);
				Py_DECREF(output);
				PyErr_SetString(PyExc_RuntimeError, exception.message());
				return 0;
			}
		} else {
			input = PyArray_FROM_OTF(input, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

			if(!input) {
				Py_XDECREF(input);
				PyErr_SetString(PyExc_TypeError, "Input should be of type `ndarray`.");
			}

			try {
				PyObject* inputObj = PyArray_FromMatrixXd(
					self->preconditioner->inverse(PyArray_ToMatrixXd(input)));
				Py_DECREF(input);
				return inputObj;

			} catch(Exception exception) {
				Py_DECREF(input);
				PyErr_SetString(PyExc_RuntimeError, exception.message());
				return 0;
			}
		}
	return 0;
}



const char* Preconditioner_logjacobian_doc =
	"loglikelihood(self, input, output)\n"
	"\n"
	"Computes the conditional log-Jacobian determinant for the given data points "
	"(using the natural logarithm).\n"
	"\n"
	"@type  input: C{ndarray}\n"
	"@param input: inputs stored in columns\n"
	"\n"
	"@type  output: C{ndarray}\n"
	"@param output: outputs stored in columns\n"
	"\n"
	"@rtype: C{ndarray}\n"
	"@return: log-Jacobian of the transformation for each data point";

PyObject* Preconditioner_logjacobian(PreconditionerObject* self, PyObject* args, PyObject* kwds) {
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
			self->preconditioner->logJacobian(PyArray_ToMatrixXd(input), PyArray_ToMatrixXd(output)));
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



PyObject* Preconditioner_new(PyTypeObject* type, PyObject*, PyObject*) {
	PyObject* self = type->tp_alloc(type, 0);

	if(self) {
		reinterpret_cast<PreconditionerObject*>(self)->preconditioner = 0;
		reinterpret_cast<PreconditionerObject*>(self)->owner = true;
	}

	return self;
}



const char* Preconditioner_doc =
	"Abstract base class for preconditioners of inputs and outputs.\n";

int Preconditioner_init(WhiteningPreconditionerObject* self, PyObject* args, PyObject* kwds) {
	PyErr_SetString(PyExc_NotImplementedError, "This is an abstract class.");
	return -1;
}



void Preconditioner_dealloc(PreconditionerObject* self) {
	if(self->preconditioner && self->owner)
		// delete actual instance
		delete self->preconditioner;

	// delete Python object
	self->ob_type->tp_free(reinterpret_cast<PyObject*>(self));
}



PyObject* Preconditioner_dim_in(PreconditionerObject* self, void*) {
	return PyInt_FromLong(self->preconditioner->dimIn());
}



PyObject* Preconditioner_dim_in_pre(PreconditionerObject* self, void*) {
	return PyInt_FromLong(self->preconditioner->dimInPre());
}



PyObject* Preconditioner_dim_out(PreconditionerObject* self, void*) {
	return PyInt_FromLong(self->preconditioner->dimOut());
}



PyObject* Preconditioner_dim_out_pre(PreconditionerObject* self, void*) {
	return PyInt_FromLong(self->preconditioner->dimOutPre());
}



const char* AffinePreconditioner_doc =
	"Performs affine transformations on the input and the output.\n"
	"\n"
	"The transformation defined by the preconditioner is\n"
	"\n"
	"$$\\mathbf{\\hat x} = \\mathbf{P}_x(\\mathbf{x} - \\mathbf{m}_x),$$\n"
	"$$\\mathbf{\\hat y} = \\mathbf{P}_y(\\mathbf{y} - \\mathbf{m}_y - \\mathbf{A}\\mathbf{\\hat x}),$$\n"
	"\n"
	"where $\\mathbf{x}$ represents the input and $\\mathbf{y}$ represents the output.\n"
	"\n"
	"@type  mean_in: C{ndarray}\n"
	"@param mean_in: a column vector which will be subtracted from the input ($\\mathbf{m}_x$)\n"
	"\n"
	"@type  mean_out: C{ndarray}\n"
	"@param mean_out: a column vector which will be subtracted from the output ($\\mathbf{m}_y$)\n"
	"\n"
	"@type  pre_in: C{ndarray}\n"
	"@param pre_in: a preconditioner for the input ($\\mathbf{P}_x$)\n"
	"\n"
	"@type  pre_out: C{ndarray}\n"
	"@param pre_out: a preconditioner for the output ($\\mathbf{P}_y$)\n"
	"\n"
	"@type  predictor: C{ndarray}\n"
	"@param predictor: a linear predictor of the output ($\\mathbf{A}$)";

int AffinePreconditioner_init(AffinePreconditionerObject* self, PyObject* args, PyObject* kwds) {
	PyObject* meanIn;
	PyObject* meanOut;
	PyObject* preIn;
	PyObject* preInInv;
	PyObject* preOut;
	PyObject* preOutInv;
	PyObject* predictor;

	// test if this call to __init__ is the result of unpickling
	if(PyArg_ParseTuple(args, "OOOOOOO", &meanIn, &meanOut, &preIn, &preInInv, &preOut, &preOutInv, &predictor)) {
		meanIn = PyArray_FROM_OTF(meanIn, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
		meanOut = PyArray_FROM_OTF(meanOut, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
		preIn = PyArray_FROM_OTF(preIn, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
		preInInv = PyArray_FROM_OTF(preInInv, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
		preOut = PyArray_FROM_OTF(preOut, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
		preOutInv = PyArray_FROM_OTF(preOutInv, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
		predictor = PyArray_FROM_OTF(predictor, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

		if(!meanIn || !meanOut || !preIn || !preInInv || !preOut || !preOutInv || !predictor) {
			Py_XDECREF(meanIn);
			Py_XDECREF(meanOut);
			Py_XDECREF(preIn);
			Py_XDECREF(preInInv);
			Py_XDECREF(preOut);
			Py_XDECREF(preOutInv);
			Py_XDECREF(predictor);
			PyErr_SetString(PyExc_TypeError, "Parameters of preconditioner should be of type `ndarray`.");
			return -1;
		}

		try {
			self->preconditioner = new AffinePreconditioner(
				PyArray_ToMatrixXd(meanIn),
				PyArray_ToMatrixXd(meanOut),
				PyArray_ToMatrixXd(preIn),
				PyArray_ToMatrixXd(preInInv),
				PyArray_ToMatrixXd(preOut),
				PyArray_ToMatrixXd(preOutInv),
				PyArray_ToMatrixXd(predictor));
		} catch(Exception exception) {
			PyErr_SetString(PyExc_RuntimeError, exception.message());
			Py_DECREF(meanIn);
			Py_DECREF(meanOut);
			Py_DECREF(preIn);
			Py_DECREF(preInInv);
			Py_DECREF(preOut);
			Py_DECREF(preOutInv);
			Py_DECREF(predictor);
			return -1;
		}

		Py_DECREF(meanIn);
		Py_DECREF(meanOut);
		Py_DECREF(preIn);
		Py_DECREF(preInInv);
		Py_DECREF(preOut);
		Py_DECREF(preOutInv);
		Py_DECREF(predictor);
	} else {
		PyErr_Clear();

		const char* kwlist[] = {"mean_in", "mean_out", "pre_in", "pre_out", "predictor", 0};

		if(!PyArg_ParseTupleAndKeywords(args, kwds, "OOOOO", const_cast<char**>(kwlist),
			&meanIn,
			&meanOut,
			&preIn,
			&preOut,
			&predictor)) 
		{
			return -1;
		}

		meanIn = PyArray_FROM_OTF(meanIn, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
		meanOut = PyArray_FROM_OTF(meanOut, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
		preIn = PyArray_FROM_OTF(preIn, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
		preOut = PyArray_FROM_OTF(preOut, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
		predictor = PyArray_FROM_OTF(predictor, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

		if(!meanIn || !meanOut || !preIn || !preOut || !predictor) {
			Py_XDECREF(meanIn);
			Py_XDECREF(meanOut);
			Py_XDECREF(preIn);
			Py_XDECREF(preOut);
			Py_XDECREF(predictor);
			PyErr_SetString(PyExc_TypeError, "All parameters should be of type `ndarray`.");
			return -1;
		}

		try {
			self->preconditioner = new AffinePreconditioner(
				PyArray_ToMatrixXd(meanIn),
				PyArray_ToMatrixXd(meanOut),
				PyArray_ToMatrixXd(preIn),
				PyArray_ToMatrixXd(preOut),
				PyArray_ToMatrixXd(predictor));
		} catch(Exception exception) {
			Py_DECREF(meanIn);
			Py_DECREF(meanOut);
			Py_DECREF(preIn);
			Py_DECREF(preOut);
			Py_DECREF(predictor);
			PyErr_SetString(PyExc_RuntimeError, exception.message());
			return -1;
		}

		Py_DECREF(meanIn);
		Py_DECREF(meanOut);
		Py_DECREF(preIn);
		Py_DECREF(preOut);
		Py_DECREF(predictor);
	}

	return 0;
}



const char* AffineTransform_doc =
	"Performs an affine transformation on inputs.\n"
	"\n"
	"$$\\mathbf{\\hat x} = \\mathbf{P}_x(\\mathbf{x} - \\mathbf{m}_x),$$\n"
	"\n"
	"Unlike L{AffinePreconditioner}, this class does not touch the output and can thus\n"
	"be used with discrete data, for example.\n"
	"\n"
	"@type  mean_in: C{ndarray}\n"
	"@param mean_in: a column vector which will be subtracted from the input ($\\mathbf{m}_x$)\n"
	"\n"
	"@type  pre_in: C{ndarray}\n"
	"@param pre_in: a preconditioner for the input ($\\mathbf{P}_x$)\n"
	"\n"
	"@type  dim_out: C{int}\n"
	"@param dim_out: dimensionality of the output (default: 1)";

int AffineTransform_init(AffineTransformObject* self, PyObject* args, PyObject* kwds) {
	PyObject* meanIn;
	PyObject* preIn;
	PyObject* preInInv;
	int dimOut = 1;

	// test if this call to __init__ is the result of unpickling
	if(PyArg_ParseTuple(args, "OOOi", &meanIn, &preIn, &preInInv, &dimOut)) {
		meanIn = PyArray_FROM_OTF(meanIn, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
		preIn = PyArray_FROM_OTF(preIn, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
		preInInv = PyArray_FROM_OTF(preInInv, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

		if(!meanIn || !preIn || !preInInv) {
			Py_XDECREF(meanIn);
			Py_XDECREF(preIn);
			Py_XDECREF(preInInv);
			PyErr_SetString(PyExc_TypeError, "Parameters of transform should be of type `ndarray`.");
			return -1;
		}

		try {
			self->preconditioner = new AffineTransform(
				PyArray_ToMatrixXd(meanIn),
				PyArray_ToMatrixXd(preIn),
				PyArray_ToMatrixXd(preInInv),
				dimOut);
		} catch(Exception exception) {
			PyErr_SetString(PyExc_RuntimeError, exception.message());
			Py_DECREF(meanIn);
			Py_DECREF(preIn);
			Py_DECREF(preInInv);
			return -1;
		}

		Py_DECREF(meanIn);
		Py_DECREF(preIn);
		Py_DECREF(preInInv);
	} else {
		PyErr_Clear();

		const char* kwlist[] = {"mean_in", "pre_in", "dim_out", 0};

		if(!PyArg_ParseTupleAndKeywords(args, kwds, "OO|i", const_cast<char**>(kwlist),
			&meanIn, &preIn, &dimOut))
		{
			return -1;
		}

		meanIn = PyArray_FROM_OTF(meanIn, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
		preIn = PyArray_FROM_OTF(preIn, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

		if(!meanIn || !preIn) {
			Py_XDECREF(meanIn);
			Py_XDECREF(preIn);
			PyErr_SetString(PyExc_TypeError, "Parameters of transform should be of type `ndarray`.");
			return -1;
		}

		try {
			self->preconditioner = new AffineTransform(
				PyArray_ToMatrixXd(meanIn),
				PyArray_ToMatrixXd(preIn),
				dimOut);
		} catch(Exception exception) {
			Py_DECREF(meanIn);
			Py_DECREF(preIn);
			PyErr_SetString(PyExc_RuntimeError, exception.message());
			return -1;
		}

		Py_DECREF(meanIn);
		Py_DECREF(preIn);
	}

	return 0;
}



PyObject* AffinePreconditioner_mean_in(AffinePreconditionerObject* self, void*) {
	return PyArray_FromMatrixXd(self->preconditioner->meanIn());
}



PyObject* AffinePreconditioner_mean_out(AffinePreconditionerObject* self, void*) {
	return PyArray_FromMatrixXd(self->preconditioner->meanOut());
}



PyObject* AffinePreconditioner_pre_in(AffinePreconditionerObject* self, void*) {
	return PyArray_FromMatrixXd(self->preconditioner->preIn());
}



PyObject* AffinePreconditioner_pre_out(AffinePreconditionerObject* self, void*) {
	return PyArray_FromMatrixXd(self->preconditioner->preOut());
}



PyObject* AffinePreconditioner_predictor(AffinePreconditionerObject* self, void*) {
	return PyArray_FromMatrixXd(self->preconditioner->predictor());
}



PyObject* AffinePreconditioner_reduce(AffinePreconditionerObject* self, PyObject*) {
	PyObject* meanIn = PyArray_FromMatrixXd(self->preconditioner->meanIn());
	PyObject* meanOut = PyArray_FromMatrixXd(self->preconditioner->meanOut());
	PyObject* preIn = PyArray_FromMatrixXd(self->preconditioner->preIn());
	PyObject* preInInv = PyArray_FromMatrixXd(self->preconditioner->preInInv());
	PyObject* preOut = PyArray_FromMatrixXd(self->preconditioner->preOut());
	PyObject* preOutInv = PyArray_FromMatrixXd(self->preconditioner->preOutInv());
	PyObject* predictor = PyArray_FromMatrixXd(self->preconditioner->predictor());

	PyObject* args = Py_BuildValue("(OOOOOOO)", 
		meanIn,
		meanOut,
		preIn,
		preInInv,
		preOut,
		preOutInv,
		predictor);
	PyObject* state = Py_BuildValue("()");
	PyObject* result = Py_BuildValue("(OOO)", Py_TYPE(self), args, state);

	Py_DECREF(meanIn);
	Py_DECREF(meanOut);
	Py_DECREF(preIn);
	Py_DECREF(preInInv);
	Py_DECREF(preOut);
	Py_DECREF(preOutInv);
	Py_DECREF(predictor);
	Py_DECREF(args);
	Py_DECREF(state);

	return result;
}



PyObject* AffinePreconditioner_setstate(AffinePreconditionerObject* self, PyObject* state) {
	// AffinePreconditioner_init does everything
	Py_INCREF(Py_None);
	return Py_None;
}



PyObject* AffineTransform_reduce(AffineTransformObject* self, PyObject*) {
	PyObject* meanIn = PyArray_FromMatrixXd(self->preconditioner->meanIn());
	PyObject* preIn = PyArray_FromMatrixXd(self->preconditioner->preIn());
	PyObject* preInInv = PyArray_FromMatrixXd(self->preconditioner->preInInv());
	int dimOut = self->preconditioner->dimOut();

	PyObject* args = Py_BuildValue("(OOOi)", 
		meanIn,
		preIn,
		preInInv,
		dimOut);
	PyObject* state = Py_BuildValue("()");
	PyObject* result = Py_BuildValue("(OOO)", Py_TYPE(self), args, state);

	Py_DECREF(meanIn);
	Py_DECREF(preIn);
	Py_DECREF(preInInv);
	Py_DECREF(args);
	Py_DECREF(state);

	return result;
}



const char* WhiteningPreconditioner_doc =
	"Decorrelates inputs and outputs.\n"
	"\n"
	"@type  input: C{ndarray}\n"
	"@param input: inputs stored in columns\n"
	"\n"
	"@type  output: C{ndarray}\n"
	"@param output: outputs stored in columns";

int WhiteningPreconditioner_init(WhiteningPreconditionerObject* self, PyObject* args, PyObject* kwds) {
	PyObject* meanIn;
	PyObject* meanOut;
	PyObject* preIn;
	PyObject* preInInv;
	PyObject* preOut;
	PyObject* preOutInv;
	PyObject* predictor;

	// test if this call to __init__ is the result of unpickling
	if(PyArg_ParseTuple(args, "OOOOOOO", &meanIn, &meanOut, &preIn, &preInInv, &preOut, &preOutInv, &predictor)) {
		meanIn = PyArray_FROM_OTF(meanIn, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
		meanOut = PyArray_FROM_OTF(meanOut, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
		preIn = PyArray_FROM_OTF(preIn, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
		preInInv = PyArray_FROM_OTF(preInInv, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
		preOut = PyArray_FROM_OTF(preOut, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
		preOutInv = PyArray_FROM_OTF(preOutInv, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
		predictor = PyArray_FROM_OTF(predictor, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

		if(!meanIn || !meanOut || !preIn || !preInInv || !preOut || !preOutInv || !predictor) {
			Py_XDECREF(meanIn);
			Py_XDECREF(meanOut);
			Py_XDECREF(preIn);
			Py_XDECREF(preInInv);
			Py_XDECREF(preOut);
			Py_XDECREF(preOutInv);
			Py_XDECREF(predictor);
			PyErr_SetString(PyExc_TypeError, "Parameters of preconditioner should be of type `ndarray`.");
			return -1;
		}

		try {
			self->preconditioner = new WhiteningPreconditioner(
				PyArray_ToMatrixXd(meanIn),
				PyArray_ToMatrixXd(meanOut),
				PyArray_ToMatrixXd(preIn),
				PyArray_ToMatrixXd(preInInv),
				PyArray_ToMatrixXd(preOut),
				PyArray_ToMatrixXd(preOutInv),
				PyArray_ToMatrixXd(predictor));
		} catch(Exception exception) {
			PyErr_SetString(PyExc_RuntimeError, exception.message());
			Py_DECREF(meanIn);
			Py_DECREF(meanOut);
			Py_DECREF(preIn);
			Py_DECREF(preInInv);
			Py_DECREF(preOut);
			Py_DECREF(preOutInv);
			Py_DECREF(predictor);
			return -1;
		}

		Py_DECREF(meanIn);
		Py_DECREF(meanOut);
		Py_DECREF(preIn);
		Py_DECREF(preInInv);
		Py_DECREF(preOut);
		Py_DECREF(preOutInv);
		Py_DECREF(predictor);
	} else {
		PyErr_Clear();

		const char* kwlist[] = {"input", "output", 0};

		PyObject* input;
		PyObject* output;


		if(!PyArg_ParseTupleAndKeywords(args, kwds, "OO", const_cast<char**>(kwlist), &input, &output)) {
			return -1;
		}

		input = PyArray_FROM_OTF(input, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
		output = PyArray_FROM_OTF(output, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

		if(!input || !output) {
			Py_XDECREF(input);
			Py_XDECREF(output);
			PyErr_SetString(PyExc_TypeError, "Input and output should be of type `ndarray`.");
			return -1;
		}

		try {
			self->preconditioner = new WhiteningPreconditioner(
				PyArray_ToMatrixXd(input),
				PyArray_ToMatrixXd(output));
		} catch(Exception& exception) {
			Py_DECREF(input);
			Py_DECREF(output);
			PyErr_SetString(PyExc_RuntimeError, exception.message());
			return -1;
		} catch(bad_alloc&) {
			Py_DECREF(input);
			Py_DECREF(output);
			PyErr_SetString(PyExc_RuntimeError, "Could not allocate memory.");
			return -1;
		}


		Py_DECREF(input);
		Py_DECREF(output);

	}

	return 0;
}



const char* WhiteningTransform_doc =
	"Decorrelates inputs.\n"
	"\n"
	"@type  input: C{ndarray}\n"
	"@param input: inputs stored in columns\n"
	"\n"
	"@type  output: C{ndarray}\n"
	"@param output: outputs stored in columns (optional)\n"
	"\n"
	"@type  dim_out: C{int}\n"
	"@param dim_out: dimensionality of the output (default: 1)";

int WhiteningTransform_init(WhiteningTransformObject* self, PyObject* args, PyObject* kwds) {
	PyObject* meanIn;
	PyObject* preIn;
	PyObject* preInInv;
	int dimOut = 1;

	// test if this call to __init__ is the result of unpickling
	if(PyArg_ParseTuple(args, "OOOi", &meanIn, &preIn, &preInInv, &dimOut)) {
		meanIn = PyArray_FROM_OTF(meanIn, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
		preIn = PyArray_FROM_OTF(preIn, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
		preInInv = PyArray_FROM_OTF(preInInv, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

		if(!meanIn || !preIn || !preInInv) {
			Py_XDECREF(meanIn);
			Py_XDECREF(preIn);
			Py_XDECREF(preInInv);
			PyErr_SetString(PyExc_TypeError, "Parameters of transform should be of type `ndarray`.");
			return -1;
		}

		try {
			self->preconditioner = new WhiteningTransform(
				PyArray_ToMatrixXd(meanIn),
				PyArray_ToMatrixXd(preIn),
				PyArray_ToMatrixXd(preInInv),
				dimOut);
		} catch(Exception exception) {
			PyErr_SetString(PyExc_RuntimeError, exception.message());
			Py_DECREF(meanIn);
			Py_DECREF(preIn);
			Py_DECREF(preInInv);
			return -1;
		}

		Py_DECREF(meanIn);
		Py_DECREF(preIn);
		Py_DECREF(preInInv);
	} else {
		PyErr_Clear();

		const char* kwlist[] = {"input", "output", "dim_out", 0};

		PyObject* input;
		PyObject* output = 0;
		dimOut = 1;

		if(!PyArg_ParseTupleAndKeywords(args, kwds, "O|Oi", const_cast<char**>(kwlist),
			&input, &output, &dimOut))
		{
			return -1;
		}

		input = PyArray_FROM_OTF(input, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

		if(!input) {
			PyErr_SetString(PyExc_TypeError, "Input should be of type `ndarray`.");
			return -1;
		}

		if(output) {
			output = PyArray_FROM_OTF(output, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

			if(!output) {
				Py_DECREF(input);
				PyErr_SetString(PyExc_TypeError, "Output should be of type `ndarray`.");
				return -1;
			}
		}

		try {
			if(output)
				self->preconditioner = new WhiteningTransform(
					PyArray_ToMatrixXd(input),
					PyArray_ToMatrixXd(output));
			else
				self->preconditioner = new WhiteningTransform(
					PyArray_ToMatrixXd(input),
					dimOut);
		} catch(Exception exception) {
			Py_DECREF(input);
			Py_XDECREF(output);
			PyErr_SetString(PyExc_RuntimeError, exception.message());
			return -1;
		}

		Py_DECREF(input);
		Py_XDECREF(output);
	}

	return 0;
}



const char* PCAPreconditioner_doc =
	"This preconditioner can be used to reduce the dimensionality of the input.\n"
	"\n"
	"Similar to L{WhiteningPreconditioner}, the transformed data will be decorrelated.\n"
	"\n"
	"To create a preconditioner which retains (at least) 98.5% of the input variance, use:\n"
	"\n"
	"\t>>> pca = PCAPreconditioner(input, output, var_explained=98.5)\n"
	"\n"
	"To create a preconditioner which reduces the dimensionality of the input to 10, use:\n"
	"\n"
	"\t>>> pca = PCAPreconditioner(input, output, num_pcs=10)\n"
	"\n"
	"If both arguments are specified, C{var_explained} will be ignored."
	"Afterwards, apply the preconditioner to the data.\n"
	"\n"
	"\t>>> input, output = preconditioner(input, output)\n"
	"\n"
	"To (approximately) reconstruct the data, you can do the following.\n"
	"\n"
	"\t>>> input, output = preconditioner.inverse(input, output)\n"
	"\n"
	"@type  input: C{ndarray}\n"
	"@param input: inputs stored in columns\n"
	"\n"
	"@type  output: C{ndarray}\n"
	"@param output: outputs stored in columns\n"
	"\n"
	"@type  var_explained: C{double}\n"
	"@param var_explained: the amount of variance retained after dimensionality reduction (in percent)\n"
	"\n"
	"@type  num_pcs: C{int}\n"
	"@param num_pcs: the number of principal components of the input kept";

int PCAPreconditioner_init(PCAPreconditionerObject* self, PyObject* args, PyObject* kwds) {
	PyObject* eigenvalues;
	PyObject* meanIn;
	PyObject* meanOut;
	PyObject* preIn;
	PyObject* preInInv;
	PyObject* preOut;
	PyObject* preOutInv;
	PyObject* predictor;

	// test if this call to __init__ is the result of unpickling
	if(PyArg_ParseTuple(args, "OOOOOOOO",
		&eigenvalues, &meanIn, &meanOut, &preIn,
		&preInInv, &preOut, &preOutInv, &predictor))
	{
		eigenvalues = PyArray_FROM_OTF(eigenvalues, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
		meanIn = PyArray_FROM_OTF(meanIn, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
		meanOut = PyArray_FROM_OTF(meanOut, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
		preIn = PyArray_FROM_OTF(preIn, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
		preInInv = PyArray_FROM_OTF(preInInv, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
		preOut = PyArray_FROM_OTF(preOut, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
		preOutInv = PyArray_FROM_OTF(preOutInv, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
		predictor = PyArray_FROM_OTF(predictor, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

		if(!eigenvalues || !meanIn || !meanOut || !preIn || !preInInv || !preOut || !preOutInv || !predictor) {
			Py_XDECREF(eigenvalues);
			Py_XDECREF(meanIn);
			Py_XDECREF(meanOut);
			Py_XDECREF(preIn);
			Py_XDECREF(preInInv);
			Py_XDECREF(preOut);
			Py_XDECREF(preOutInv);
			Py_XDECREF(predictor);
			PyErr_SetString(PyExc_TypeError, "Parameters of preconditioner should be of type `ndarray`.");
			return -1;
		}

		try {
			self->preconditioner = new PCAPreconditioner(
				PyArray_ToMatrixXd(eigenvalues),
				PyArray_ToMatrixXd(meanIn),
				PyArray_ToMatrixXd(meanOut),
				PyArray_ToMatrixXd(preIn),
				PyArray_ToMatrixXd(preInInv),
				PyArray_ToMatrixXd(preOut),
				PyArray_ToMatrixXd(preOutInv),
				PyArray_ToMatrixXd(predictor));
		} catch(Exception exception) {
			PyErr_SetString(PyExc_RuntimeError, exception.message());
			Py_DECREF(eigenvalues);
			Py_DECREF(meanIn);
			Py_DECREF(meanOut);
			Py_DECREF(preIn);
			Py_DECREF(preInInv);
			Py_DECREF(preOut);
			Py_DECREF(preOutInv);
			Py_DECREF(predictor);
			return -1;
		}

		Py_DECREF(eigenvalues);
		Py_DECREF(meanIn);
		Py_DECREF(meanOut);
		Py_DECREF(preIn);
		Py_DECREF(preInInv);
		Py_DECREF(preOut);
		Py_DECREF(preOutInv);
		Py_DECREF(predictor);
	} else {
		PyErr_Clear();

		const char* kwlist[] = {"input", "output", "var_explained", "num_pcs", 0};

		PyObject* input;
		PyObject* output;
		double var_explained = 99.;
		int num_pcs = -1;

		if(!PyArg_ParseTupleAndKeywords(args, kwds, "OO|di", const_cast<char**>(kwlist),
			&input, &output, &var_explained, &num_pcs))
		{
			return -1;
		}

		input = PyArray_FROM_OTF(input, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
		output = PyArray_FROM_OTF(output, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

		if(!input || !output) {
			Py_XDECREF(input);
			Py_XDECREF(output);
			PyErr_SetString(PyExc_TypeError, "Input and output should be of type `ndarray`.");
			return -1;
		}

		try {
			self->preconditioner = new PCAPreconditioner(
				PyArray_ToMatrixXd(input),
				PyArray_ToMatrixXd(output),
				var_explained,
				num_pcs);
		} catch(Exception& exception) {
			Py_DECREF(input);
			Py_DECREF(output);
			PyErr_SetString(PyExc_RuntimeError, exception.message());
			return -1;
		} catch(bad_alloc&) {
			Py_DECREF(input);
			Py_DECREF(output);
			PyErr_SetString(PyExc_RuntimeError, "Could not allocate memory.");
			return -1;
		}

		Py_DECREF(input);
		Py_DECREF(output);

	}

	return 0;
}



const char* PCATransform_doc =
	"This class behaves like L{PCAPreconditioner}, but does not change the output.\n"
	"\n"
	"@type  input: C{ndarray}\n"
	"@param input: inputs stored in columns\n"
	"\n"
	"@type  output: C{ndarray}\n"
	"@param output: outputs stored in columns\n"
	"\n"
	"@type  dim_out: C{int}\n"
	"@param dim_out: number of outputs (default: 1)\n"
	"\n"
	"@type  var_explained: C{double}\n"
	"@param var_explained: the amount of variance retained after dimensionality reduction (in percent)\n"
	"\n"
	"@type  num_pcs: C{int}\n"
	"@param num_pcs: the number of principal components of the input kept";

int PCATransform_init(PCATransformObject* self, PyObject* args, PyObject* kwds) {
	PyObject* eigenvalues;
	PyObject* meanIn;
	PyObject* preIn;
	PyObject* preInInv;
	int dimOut = 1;

	// test if this call to __init__ is the result of unpickling
	if(PyArg_ParseTuple(args, "OOOOi", &eigenvalues, &meanIn, &preIn, &preInInv, &dimOut) && PyArray_Check(preIn)) {
		eigenvalues = PyArray_FROM_OTF(eigenvalues, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
		meanIn = PyArray_FROM_OTF(meanIn, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
		preIn = PyArray_FROM_OTF(preIn, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
		preInInv = PyArray_FROM_OTF(preInInv, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

		if(!eigenvalues || !meanIn || !preIn || !preInInv) {
			Py_XDECREF(eigenvalues);
			Py_XDECREF(meanIn);
			Py_XDECREF(preIn);
			Py_XDECREF(preInInv);
			PyErr_SetString(PyExc_TypeError, "Parameters of transform should be of type `ndarray`.");
			return -1;
		}

		try {
			self->preconditioner = new PCATransform(
				PyArray_ToMatrixXd(eigenvalues),
				PyArray_ToMatrixXd(meanIn),
				PyArray_ToMatrixXd(preIn),
				PyArray_ToMatrixXd(preInInv),
				dimOut);
		} catch(Exception exception) {
			PyErr_SetString(PyExc_RuntimeError, exception.message());
			Py_DECREF(eigenvalues);
			Py_DECREF(meanIn);
			Py_DECREF(preIn);
			Py_DECREF(preInInv);
			return -1;
		}

		Py_DECREF(eigenvalues);
		Py_DECREF(meanIn);
		Py_DECREF(preIn);
		Py_DECREF(preInInv);
	} else {
		PyErr_Clear();

		const char* kwlist[] = {"input", "output", "var_explained", "num_pcs", "dim_out", 0};

		PyObject* input;
		PyObject* output = 0;
		double var_explained = 99.;
		int num_pcs = -1;

		if(!PyArg_ParseTupleAndKeywords(args, kwds, "O|Odii", const_cast<char**>(kwlist),
			&input, &output, &var_explained, &num_pcs, &dimOut))
		{
			return -1;
		}

		input = PyArray_FROM_OTF(input, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

		if(!input) {
			PyErr_SetString(PyExc_TypeError, "Input should be of type `ndarray`.");
			return -1;
		}

		if(output) {
			output = PyArray_FROM_OTF(output, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

			if(!output) {
				PyErr_SetString(PyExc_TypeError, "Output should be of type `ndarray`.");
				return -1;
			}
		}

		try {
			if(output)
				self->preconditioner = new PCATransform(
					PyArray_ToMatrixXd(input),
					PyArray_ToMatrixXd(output),
					var_explained,
					num_pcs);
			else
				self->preconditioner = new PCATransform(
					PyArray_ToMatrixXd(input),
					var_explained,
					num_pcs,
					dimOut);
		} catch(Exception& exception) {
			Py_DECREF(input);
			Py_XDECREF(output);
			PyErr_SetString(PyExc_RuntimeError, exception.message());
			return -1;
		} catch(bad_alloc&) {
			Py_DECREF(input);
			Py_XDECREF(output);
			PyErr_SetString(PyExc_RuntimeError, "Could not allocate memory.");
			return -1;
		}

		Py_DECREF(input);
		Py_XDECREF(output);
	}

	return 0;
}



PyObject* PCAPreconditioner_eigenvalues(PCAPreconditionerObject* self, void*) {
	return PyArray_FromMatrixXd(self->preconditioner->eigenvalues());
}



PyObject* PCATransform_eigenvalues(PCATransformObject* self, void*) {
	return PyArray_FromMatrixXd(self->preconditioner->eigenvalues());
}



PyObject* PCAPreconditioner_reduce(PCAPreconditionerObject* self, PyObject*) {
	PyObject* eigenvalues = PyArray_FromMatrixXd(self->preconditioner->eigenvalues());
	PyObject* meanIn = PyArray_FromMatrixXd(self->preconditioner->meanIn());
	PyObject* meanOut = PyArray_FromMatrixXd(self->preconditioner->meanOut());
	PyObject* preIn = PyArray_FromMatrixXd(self->preconditioner->preIn());
	PyObject* preInInv = PyArray_FromMatrixXd(self->preconditioner->preInInv());
	PyObject* preOut = PyArray_FromMatrixXd(self->preconditioner->preOut());
	PyObject* preOutInv = PyArray_FromMatrixXd(self->preconditioner->preOutInv());
	PyObject* predictor = PyArray_FromMatrixXd(self->preconditioner->predictor());

	PyObject* args = Py_BuildValue("(OOOOOOOO)",
		eigenvalues,
		meanIn,
		meanOut,
		preIn,
		preInInv,
		preOut,
		preOutInv,
		predictor);
	PyObject* state = Py_BuildValue("()");
	PyObject* result = Py_BuildValue("(OOO)", Py_TYPE(self), args, state);

	Py_DECREF(eigenvalues);
	Py_DECREF(meanIn);
	Py_DECREF(meanOut);
	Py_DECREF(preIn);
	Py_DECREF(preInInv);
	Py_DECREF(preOut);
	Py_DECREF(preOutInv);
	Py_DECREF(predictor);
	Py_DECREF(args);
	Py_DECREF(state);

	return result;
}



PyObject* PCATransform_reduce(PCATransformObject* self, PyObject*) {
	PyObject* eigenvalues = PyArray_FromMatrixXd(self->preconditioner->eigenvalues());
	PyObject* meanIn = PyArray_FromMatrixXd(self->preconditioner->meanIn());
	PyObject* preIn = PyArray_FromMatrixXd(self->preconditioner->preIn());
	PyObject* preInInv = PyArray_FromMatrixXd(self->preconditioner->preInInv());
	int dimOut = self->preconditioner->dimOut();

	PyObject* args = Py_BuildValue("(OOOOi)",
		eigenvalues,
		meanIn,
		preIn,
		preInInv,
		dimOut);
	PyObject* state = Py_BuildValue("()");
	PyObject* result = Py_BuildValue("(OOO)", Py_TYPE(self), args, state);

	Py_DECREF(eigenvalues);
	Py_DECREF(meanIn);
	Py_DECREF(preIn);
	Py_DECREF(preInInv);
	Py_DECREF(args);
	Py_DECREF(state);

	return result;
}



const char* BinningTransform_doc =
	"Sums up neighboring dimensions of input vectors.\n"
	"\n"
	"@type  binning: C{int}\n"
	"@param binning: bin width\n"
	"\n"
	"@type  dim_in: C{int}\n"
	"@param dim_in: dimensionality of inputs\n"
	"\n"
	"@type  dim_out: C{int}\n"
	"@param dim_out: dimensionality of outputs (default: 1)\n";

int BinningTransform_init(BinningTransformObject* self, PyObject* args, PyObject* kwds) {
	// test if this call to __init__ is the result of unpickling
	const char* kwlist[] = {"binning", "dim_in", "dim_out", 0};

	int binning;
	int dim_in;
	int dim_out = 1;


	if(!PyArg_ParseTupleAndKeywords(args, kwds, "ii|i", const_cast<char**>(kwlist),
		&binning, &dim_in, &dim_out))
	{
		return -1;
	}

	try {
		self->preconditioner = new BinningTransform(binning, dim_in, dim_out);
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	return 0;
}



PyObject* BinningTransform_binning(BinningTransformObject* self, void*) {
	return PyInt_FromLong(self->preconditioner->binning());
}



PyObject* BinningTransform_reduce(BinningTransformObject* self, PyObject*) {
	PyObject* args = Py_BuildValue("(iii)",
		self->preconditioner->binning(),
		self->preconditioner->dimIn(),
		self->preconditioner->dimOut());
	PyObject* state = Py_BuildValue("()");
	PyObject* result = Py_BuildValue("(OOO)", Py_TYPE(self), args, state);

	return result;
}
