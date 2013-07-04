#include "conditionaldistributioninterface.h"
#include "preconditionerinterface.h"
#include "Eigen/Core"

#include "cmt/utils"
using CMT::Exception;

#include "cmt/models"
using CMT::ConditionalDistribution;

PyObject* CD_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
	PyObject* self = type->tp_alloc(type, 0);

	if(self) {
		reinterpret_cast<CDObject*>(self)->cd = 0;
		reinterpret_cast<CDObject*>(self)->owner = true;
	}

	return self;
}



const char* CD_doc =
	"Abstract base class for conditional models.\n";

int CD_init(CDObject* self, PyObject* args, PyObject* kwds) {
	PyErr_SetString(PyExc_NotImplementedError, "This is an abstract class.");
	return -1;
}



void CD_dealloc(CDObject* self) {
	// delete actual instance
	if(self->owner)
		delete self->cd;

	// delete CDObject
	self->ob_type->tp_free(reinterpret_cast<PyObject*>(self));
}



PyObject* CD_dim_in(CDObject* self, PyObject*, void*) {
	return PyInt_FromLong(self->cd->dimIn());
}



PyObject* CD_dim_out(CDObject* self, PyObject*, void*) {
	return PyInt_FromLong(self->cd->dimOut());
}



const char* CD_sample_doc =
	"sample(self, input)\n"
	"\n"
	"Generates outputs for given inputs.\n"
	"\n"
	"@type  input: ndarray\n"
	"@param input: inputs stored in columns\n"
	"\n"
	"@rtype: ndarray\n"
	"@return: sampled outputs";

PyObject* CD_sample(CDObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"input", 0};

	PyObject* input;

	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O", const_cast<char**>(kwlist), &input))
		return 0;

	input = PyArray_FROM_OTF(input, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

	if(!input) {
		PyErr_SetString(PyExc_TypeError, "Data has to be stored in a NumPy array.");
		return 0;
	}

	try {
		PyObject* result = PyArray_FromMatrixXd(self->cd->sample(PyArray_ToMatrixXd(input)));
		Py_DECREF(input);
		return result;
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		Py_DECREF(input);
		return 0;
	}

	return 0;
}



const char* CD_loglikelihood_doc =
	"loglikelihood(self, input, output)\n"
	"\n"
	"Computes the conditional log-likelihood for the given data points in nats.\n"
	"\n"
	"@type  input: ndarray\n"
	"@param input: inputs stored in columns\n"
	"\n"
	"@type  output: ndarray\n"
	"@param output: outputs stored in columns\n"
	"\n"
	"@rtype: ndarray\n"
	"@return: log-likelihood of the model evaluated for each data point";

PyObject* CD_loglikelihood(CDObject* self, PyObject* args, PyObject* kwds) {
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
			self->cd->logLikelihood(PyArray_ToMatrixXd(input), PyArray_ToMatrixXd(output)));
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



const char* CD_evaluate_doc =
	"evaluate(self, input, output, preconditioner=None)\n"
	"\n"
	"Computes the average negative conditional log-likelihood for the given data points "
	"in bits per output component (smaller is better).\n"
	"\n"
	"If a preconditioner is specified, the data is transformed before computing the likelihood "
	"and the result is corrected for the Jacobian of the transformation. Note that the data should "
	"*not* already be transformed when specifying a preconditioner.\n"
	"\n"
	"@type  input: ndarray\n"
	"@param input: inputs stored in columns\n"
	"\n"
	"@type  output: ndarray\n"
	"@param output: outputs stored in columns\n"
	"\n"
	"@type  preconditioner: L{Preconditioner}\n"
	"@param preconditioner: preconditioner that is used to transform the data\n"
	"\n"
	"@rtype: double\n"
	"@return: performance in bits per component";

PyObject* CD_evaluate(CDObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"input", "output", "preconditioner", 0};

	PyObject* input;
	PyObject* output;
	PyObject* preconditioner = 0;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "OO|O", const_cast<char**>(kwlist),
		&input, &output, &preconditioner))
		return 0;

	if(preconditioner == Py_None)
		return 0;

	if(preconditioner && !PyType_IsSubtype(Py_TYPE(preconditioner), &Preconditioner_type)) {
		PyErr_SetString(PyExc_TypeError, "Preconditioner has wrong type.");
		return 0;
	}

	// make sure data is stored in NumPy array
	input = PyArray_FROM_OTF(input, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
	output = PyArray_FROM_OTF(output, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

	if(!input || !output) {
		PyErr_SetString(PyExc_TypeError, "Data has to be stored in NumPy arrays.");
		return 0;
	}

	try {
		double result;

		if(preconditioner)
			result = self->cd->evaluate(
				PyArray_ToMatrixXd(input),
				PyArray_ToMatrixXd(output),
				*reinterpret_cast<PreconditionerObject*>(preconditioner)->preconditioner);
		else
			result = self->cd->evaluate(
				PyArray_ToMatrixXd(input),
				PyArray_ToMatrixXd(output));
		Py_DECREF(input);
		Py_DECREF(output);
		return PyFloat_FromDouble(result);
	} catch(Exception exception) {
		Py_DECREF(input);
		Py_DECREF(output);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	return 0;
}
