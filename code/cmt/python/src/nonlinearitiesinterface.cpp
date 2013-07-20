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
	"Abstract base class for nonlinear functions used, for example, by L{GLM}.";

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



const char* LogisticFunction_reduce_doc =
	"__reduce__(self)\n"
	"\n"
	"Method used by Pickle.";

PyObject* LogisticFunction_reduce(LogisticFunctionObject* self, PyObject*) {
	PyObject* args = Py_BuildValue("()");
	PyObject* result = Py_BuildValue("(OO)", Py_TYPE(self), args);

	Py_DECREF(args);

	return result;
}
