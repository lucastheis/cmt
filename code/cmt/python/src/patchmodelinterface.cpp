#include "patchmodelinterface.h"
#include "pyutils.h"
#include "distributioninterface.h"

#include "cmt/utils"
using CMT::Exception;

#if PY_MAJOR_VERSION >= 3
	#define PyInt_FromLong PyLong_FromLong
#endif

const char* PatchModel_doc =
	"Abstract base class for models of image patches.";

PyObject* PatchModel_rows(PatchModelObject* self, void*) {
	return PyInt_FromLong(self->distribution->rows());
}



PyObject* PatchModel_cols(PatchModelObject* self, void*) {
	return PyInt_FromLong(self->distribution->cols());
}




PyObject* PatchModel_input_mask(PatchModelObject* self, PyObject* args) {
	int i = -1;
	int j = -1;

	if(args && !PyArg_ParseTuple(args, "|ii", &i, &j))
		return 0;

	if(i >= 0 && j < 0) {
		PyErr_SetString(PyExc_TypeError, "Index should consist of a row and a column.");
		return 0;
	}

	PyObject* array;

	if(i < 0 || j < 0)
		array = PyArray_FromMatrixXb(self->distribution->inputMask());
	else
		array = PyArray_FromMatrixXb(self->distribution->inputMask(i, j));

	// make array immutable
	reinterpret_cast<PyArrayObject*>(array)->flags &= ~NPY_WRITEABLE;
	return array;
}



PyObject* PatchModel_output_mask(PatchModelObject* self, PyObject* args) {
	int i = -1;
	int j = -1;

	if(args && !PyArg_ParseTuple(args, "|ii", &i, &j))
		return 0;

	if(i >= 0 && j < 0) {
		PyErr_SetString(PyExc_TypeError, "Index should consist of a row and a column.");
		return 0;
	}

	PyObject* array;

	if(i < 0 || j < 0)
		array = PyArray_FromMatrixXb(self->distribution->outputMask());
	else
		array = PyArray_FromMatrixXb(self->distribution->outputMask(i, j));

	// make array immutable
	reinterpret_cast<PyArrayObject*>(array)->flags &= ~NPY_WRITEABLE;
	return array;
}



PyObject* PatchModel_input_indices(PatchModelObject* self, PyObject* args) {
	int i;
	int j;

	if(args && !PyArg_ParseTuple(args, "ii", &i, &j))
		return 0;

	if(i >= 0 && j < 0) {
		PyErr_SetString(PyExc_TypeError, "Index should consist of a row and a column.");
		return 0;
	}

	return PyList_FromTuples(self->distribution->inputIndices(i, j));
}



PyObject* PatchModel_order(PatchModelObject* self, void*) {
	return PyList_FromTuples(self->distribution->order());
}



PyObject* PatchModel_loglikelihood(PatchModelObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"i", "j", "data", 0};

	PyObject* data;
	int i = -1;
	int j = -1;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "iiO", const_cast<char**>(kwlist),
		&i, &j, &data))
	{
		PyErr_Clear();

		const char* kwlist[] = {"data", 0};

		if(!PyArg_ParseTupleAndKeywords(args, kwds, "O", const_cast<char**>(kwlist), &data))
			return 0;
	}

	// make sure data is stored in NumPy array
	data = PyArray_FROM_OTF(data, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

	if(!data) {
		PyErr_SetString(PyExc_TypeError, "Data has to be stored in NumPy arrays.");
		return 0;
	}

	try {
		PyObject* result;
		if(i > -1 && j > -1)
			result = PyArray_FromMatrixXd(
				self->distribution->logLikelihood(i, j, PyArray_ToMatrixXd(data)));
		else
			result = PyArray_FromMatrixXd(
				self->distribution->logLikelihood(PyArray_ToMatrixXd(data)));
		Py_DECREF(data);
		return result;
	} catch(Exception exception) {
		Py_DECREF(data);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	return 0;
}
