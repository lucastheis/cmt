#include "conditionaldistributioninterface.h"
#include "fvbninterface.h"
#include "patchmodelinterface.h"
#include "preconditionerinterface.h"
#include "distributioninterface.h"
#include "glminterface.h"

#include "exception.h"
using CMT::Exception;

GLM::Nonlinearity* fvbnNonlinearity = new LogisticFunction;
GLM::UnivariateDistribution* fvbnDistribution = new Bernoulli;

const char* FVBN_doc =
	"Model image patches using a GLM for each conditional distribution.\n"
	"\n"
	"@type  rows: integer\n"
	"@param rows: number of rows of the image patch\n"
	"\n"
	"@type  cols: integer\n"
	"@param cols: number of columns of the image patch\n"
	"\n"
	"@type  xmask: C{ndarray}\n"
	"@param xmask: a Boolean array describing the input pixels\n"
	"\n"
	"@type  ymask: C{ndarray}\n"
	"@param ymask: a Boolean array describing the output pixels\n"
	"\n"
	"@type  model: L{GLM}\n"
	"@param model: model used as a template to initialize conditional distributions\n"
	"\n"
	"@type  max_pcs: integer\n"
	"@param max_pcs: can be used to reduce dimensionality of inputs to conditional models";

int FVBN_init(FVBNObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"rows", "cols", "xmask", "ymask", "model", "max_pcs", 0};

	int rows;
	int cols;
	PyObject* xmask;
	PyObject* ymask;
	PyObject* model = 0;
	int max_pcs = -1;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "iiOO|Oi", const_cast<char**>(kwlist),
		&rows, &cols, &xmask, &ymask, &model, &max_pcs))
		return -1;

	if(model == Py_None)
		model = 0;

	if(model && !PyType_IsSubtype(Py_TYPE(model), &GLM_type)) {
		PyErr_SetString(PyExc_TypeError, "Model has to be of type `GLM`.");
		return 0;
	}

	xmask = PyArray_FROM_OTF(xmask, NPY_BOOL, NPY_F_CONTIGUOUS | NPY_ALIGNED);
	ymask = PyArray_FROM_OTF(ymask, NPY_BOOL, NPY_F_CONTIGUOUS | NPY_ALIGNED);

	if(!xmask || !ymask) {
		Py_XDECREF(xmask);
		Py_XDECREF(ymask);
		PyErr_SetString(PyExc_TypeError, "Masks have to be given as Boolean arrays.");
		return 0;
	}

	// create the actual model
	try {
		if(model) {
			self->fvbn = new PatchModel<GLM, PCATransform>(
				rows,
				cols,
				PyArray_ToMatrixXb(xmask),
				PyArray_ToMatrixXb(ymask),
				reinterpret_cast<GLMObject*>(model)->glm,
				max_pcs);
			self->nonlinearityType = Py_TYPE(reinterpret_cast<GLMObject*>(model)->nonlinearity);
			self->distributionType = Py_TYPE(reinterpret_cast<GLMObject*>(model)->distribution);
		} else {
			GLM* glm = new GLM(1, fvbnNonlinearity, fvbnDistribution);

			self->fvbn = new PatchModel<GLM, PCATransform>(
				rows,
				cols,
				PyArray_ToMatrixXb(xmask),
				PyArray_ToMatrixXb(ymask),
				glm,
				max_pcs);
			self->nonlinearityType = &LogisticFunction_type;
			self->distributionType = &Bernoulli_type;

			delete glm;
		}
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	return 0;
}



PyObject* FVBN_subscript(FVBNObject* self, PyObject* key) {
	if(!PyTuple_Check(key)) {
		PyErr_SetString(PyExc_TypeError, "Index must be a tuple.");
		return 0;
	}

	int i;
	int j;

	if(!PyArg_ParseTuple(key, "ii", &i, &j)) {
		PyErr_SetString(PyExc_TypeError, "Index should consist of a row and a column.");
		return 0;
	}

	GLM* glm = &self->fvbn->operator()(i, j);

	PyObject* nonlinearity = Nonlinearity_new(self->nonlinearityType, 0, 0);
	PyObject* distribution = Distribution_new(self->distributionType, 0, 0);

	Py_INCREF(nonlinearity);
	Py_INCREF(distribution);

	reinterpret_cast<NonlinearityObject*>(nonlinearity)->nonlinearity =
		glm->nonlinearity();
	reinterpret_cast<NonlinearityObject*>(nonlinearity)->owner = false;
	reinterpret_cast<UnivariateDistributionObject*>(distribution)->distribution =
		glm->distribution();
	reinterpret_cast<UnivariateDistributionObject*>(distribution)->owner = false;

	PyObject* glmObject = CD_new(&GLM_type, 0, 0);

	Py_INCREF(glmObject);

	reinterpret_cast<GLMObject*>(glmObject)->glm = glm;
	reinterpret_cast<GLMObject*>(glmObject)->owner = false;
	reinterpret_cast<GLMObject*>(glmObject)->nonlinearity =
		reinterpret_cast<NonlinearityObject*>(nonlinearity);
	reinterpret_cast<GLMObject*>(glmObject)->distribution =
		reinterpret_cast<UnivariateDistributionObject*>(distribution);

	return glmObject;
}



int FVBN_ass_subscript(FVBNObject* self, PyObject* key, PyObject* value) {
	if(!PyType_IsSubtype(Py_TYPE(value), &GLM_type)) {
		PyErr_SetString(PyExc_TypeError, "Conditional distribution should be a subtype of `GLM`.");
		return -1;
	}

	if(!PyTuple_Check(key)) {
		PyErr_SetString(PyExc_TypeError, "Index must be a tuple.");
		return -1;
	}

	int i;
	int j;

	if(!PyArg_ParseTuple(key, "ii", &i, &j)) {
		PyErr_SetString(PyExc_TypeError, "Index should consist of a row and a column.");
		return -1;
	}

 	self->fvbn->operator()(i, j) = *reinterpret_cast<GLMObject*>(value)->glm;

	return 0;
}



PyObject* FVBN_preconditioner(FVBNObject* self, PyObject* args) {
	int i;
	int j;

	if(!PyArg_ParseTuple(args, "ii", &i, &j)) {
		PyErr_SetString(PyExc_TypeError, "Index should consist of a row and a column.");
		return 0;
	}

	try {
		PCATransform* pc = &self->fvbn->preconditioner(i, j);
		PyObject* preconditioner = Preconditioner_new(&PCATransform_type, 0, 0);
		reinterpret_cast<PCATransformObject*>(preconditioner)->owner = false;
		reinterpret_cast<PCATransformObject*>(preconditioner)->preconditioner = pc;
		Py_INCREF(preconditioner);
		return preconditioner;
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	return 0;
}



PyObject* FVBN_preconditioners(FVBNObject* self, void*) {
	if(self->fvbn->maxPCs() < 0)
		return PyDict_New();

	PyObject* preconditioners = PyDict_New();

	for(int i = 0; i < self->fvbn->rows(); ++i) {
		for(int j = 0; j < self->fvbn->cols(); ++j) {
			PyObject* index = Py_BuildValue("(ii)", i, j);
			PyObject* preconditioner = FVBN_preconditioner(self, index);

			if(!preconditioner) {
				PyErr_Clear();
				Py_DECREF(index);
				continue;
			}

			PyDict_SetItem(preconditioners, index, preconditioner);

			Py_DECREF(index);
			Py_DECREF(preconditioner);
		}
	}

	return preconditioners;
}



int FVBN_set_preconditioners(FVBNObject* self, PyObject* value, void*) {
	if(!PyDict_Check(value)) {
		PyErr_SetString(PyExc_RuntimeError, "Preconditioners have to be stored in a dictionary."); 
		return -1;
	}

	for(int i = 0; i < self->fvbn->rows(); ++i)
		for(int j = 0; j < self->fvbn->cols(); ++j) {
			PyObject* index = Py_BuildValue("(ii)", i, j);
			PyObject* preconditioner = PyDict_GetItem(value, index);

			if(!preconditioner)
				continue;

			if(!PyType_IsSubtype(Py_TYPE(preconditioner), &PCATransform_type)) {
				PyErr_SetString(PyExc_RuntimeError,
					"All preconditioners must be of type `PCATransform`.");
				return -1;
			}

			try {
 				self->fvbn->setPreconditioner(i, j,
 					*reinterpret_cast<PCATransformObject*>(preconditioner)->preconditioner);
			} catch(Exception exception) {
				PyErr_SetString(PyExc_RuntimeError, exception.message());
				return -1;
			}

			Py_DECREF(index);
		}

	return 0;
}



const char* FVBN_initialize_doc =
	"initialize(self, data, parameters=None)\n"
	"\n"
	"Trains the model assuming shift-invariance of the patch statistics.\n"
	"\n"
	"A single conditional distribution is fitted to the given data and all models with\n"
	"a I{complete} neighborhood are initialized with this one set of parameters.\n"
	"\n"
	"It is assumed that the patches are stored in row-order ('C') in the columns of\n"
	"L{data}.\n"
	"\n"
	"@type  data: ndarray\n"
	"@param data: image patches stored column-wise";

PyObject* FVBN_initialize(FVBNObject* self, PyObject* args, PyObject* kwds) {
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
		PyErr_SetString(PyExc_TypeError, "Data has to be stored in NumPy arrays.");
		return 0;
	}

	try {
		Trainable::Parameters* params = PyObject_ToGLMParameters(parameters);

		self->fvbn->initialize(PyArray_ToMatrixXd(data), *params);

		delete params;

		Py_DECREF(data);

		Py_INCREF(Py_None);
		return Py_None;
	} catch(Exception exception) {
		Py_DECREF(data);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	return 0;
}



const char* FVBN_train_doc =
	"train(self, data, dat_val=None, parameters=None)\n"
	"\n"
	"Trains the model to the given image patches by fitting each conditional\n"
	"distribution in turn.\n"
	"\n"
	"It is assumed that the patches are stored in row-order ('C') in the columns of\n"
	"L{data}. If hyperparameters are given, they are passed on to each conditional\n"
	"distribution.\n"
	"\n"
	"@type  data: ndarray\n"
	"@param data: image patches stored column-wise\n"
	"\n"
	"@type  data_val: ndarray\n"
	"@param data_val: image patches used for early stopping based on validation error\n"
	"\n"
	"@type  parameters: dict\n"
	"@param parameters: a dictionary containing hyperparameters\n"
	"\n"
	"@rtype: bool\n"
	"@return: C{True} if training of all models converged, otherwise C{False}";

PyObject* FVBN_train(FVBNObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"i", "j", "data", "data_val", "parameters", 0};

	PyObject* data;
	PyObject* data_val = 0;
	PyObject* parameters = 0;
	int i = -1;
	int j = -1;

	// read arguments
	if(!PyArg_ParseTupleAndKeywords(args, kwds, "iiO|OO", const_cast<char**>(kwlist),
		&i, &j,
		&data,
		&data_val,
		&parameters))
	{
		PyErr_Clear();

		const char* kwlist[] = {"data", "data_val", "parameters", 0};

		if(!PyArg_ParseTupleAndKeywords(args, kwds, "O|OO", const_cast<char**>(kwlist),
			&data,
			&data_val,
			&parameters))
			return 0;
	}

	// make sure data is stored in NumPy array
	data = PyArray_FROM_OTF(data, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

	if(!data) {
		PyErr_SetString(PyExc_TypeError, "Data has to be stored in NumPy array.");
		return 0;
	}

	if(data_val == Py_None)
		data_val = 0;

	if(data_val) {
		data_val = PyArray_FROM_OTF(data_val, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

		if(!data_val) {
			Py_DECREF(data);
			PyErr_SetString(PyExc_TypeError, "Validation data has to be stored in NumPy array.");
			return 0;
		}
	}

	try {
		bool converged;

		Trainable::Parameters* params = PyObject_ToGLMParameters(parameters);

		if(data_val) {
			if(i > -1 && j > -1)
				converged = self->fvbn->train(
					i, j,
					PyArray_ToMatrixXd(data),
					PyArray_ToMatrixXd(data_val),
					*params);
			else
				converged = self->fvbn->train(
					PyArray_ToMatrixXd(data),
					PyArray_ToMatrixXd(data_val),
					*params);
		} else {
			if(i > -1 && j > -1)
				converged = self->fvbn->train(
					i, j,
					PyArray_ToMatrixXd(data),
					*params);
			else
				converged = self->fvbn->train(
					PyArray_ToMatrixXd(data),
					*params);
		}

		delete params;

		if(converged) {
			Py_DECREF(data);
			Py_XDECREF(data_val);
			Py_INCREF(Py_True);
			return Py_True;
		} else {
			Py_DECREF(data);
			Py_XDECREF(data_val);
			Py_INCREF(Py_False);
			return Py_False;
		}
	} catch(Exception exception) {
		Py_DECREF(data);
		Py_XDECREF(data_val);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	return 0;
}



const char* FVBN_reduce_doc =
	"__reduce__(self)\n"
	"\n"
	"Method used by Pickle.";

PyObject* FVBN_reduce(FVBNObject* self, PyObject*) {
	int rows = self->fvbn->rows();
	int cols = self->fvbn->cols();
	int maxPCs = self->fvbn->maxPCs();

	PyObject* inputMask = PatchModel_input_mask(reinterpret_cast<PatchModelObject*>(self), 0);
	PyObject* outputMask = PatchModel_output_mask(reinterpret_cast<PatchModelObject*>(self), 0);

	// constructor arguments
	PyObject* args = Py_BuildValue("(iiOOOi)",
		rows,
		cols,
		inputMask,
		outputMask,
		Py_None,
		maxPCs);
	
	Py_DECREF(inputMask);
	Py_DECREF(outputMask);

	// parameters
	PyObject* models = PyTuple_New(self->fvbn->dim());

	for(int i = 0; i < rows; ++i)
		for(int j = 0; j < cols; ++j) {
			PyObject* index = Py_BuildValue("(ii)", i, j);
			PyObject* glm = FVBN_subscript(self, index);

			// add GLM to list of models
			PyTuple_SetItem(models, i * cols + j, glm);

			Py_DECREF(index);
		}

	PyObject* preconditioners = FVBN_preconditioners(self, 0);

	PyObject* state = Py_BuildValue("(OO)", models, preconditioners);
	PyObject* result = Py_BuildValue("(OOO)", Py_TYPE(self), args, state);

	Py_DECREF(preconditioners);
	Py_DECREF(args);
	Py_DECREF(state);

	return result;
}



const char* FVBN_setstate_doc =
	"__setstate__(self)\n"
	"\n"
	"Method used by Pickle.";

PyObject* FVBN_setstate(FVBNObject* self, PyObject* state) {
	int cols = self->fvbn->cols();

	// for some reason the actual state is encapsulated in another tuple
	state = PyTuple_GetItem(state, 0);

	PyObject* models = PyTuple_GetItem(state, 0);
  	PyObject* preconditioners = PyTuple_GetItem(state, 1);

	if(PyTuple_Size(models) != self->fvbn->dim()) {
		PyErr_SetString(PyExc_RuntimeError, "Something went wrong while unpickling the model.");
		return 0;
	}

	try {
		for(int i = 0; i < self->fvbn->dim(); ++i) {
			PyObject* index = Py_BuildValue("(ii)", i / cols, i % cols);
			FVBN_ass_subscript(self, index, PyTuple_GetItem(models, i));
			Py_DECREF(index);
		}
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

  	if(preconditioners)
  		FVBN_set_preconditioners(self, preconditioners, 0);

	Py_INCREF(Py_None);
	return Py_None;
}
