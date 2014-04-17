#include "pyutils.h"
#include <inttypes.h>

#include "cmt/utils"
using CMT::Exception;

#include "Eigen/Core"
using Eigen::Map;
using Eigen::ColMajor;
using Eigen::RowMajor;

#include <utility>
using std::make_pair;

PyObject* PyArray_FromMatrixXd(const MatrixXd& mat) {
	// matrix dimensionality
	npy_intp dims[2];
	dims[0] = mat.rows();
	dims[1] = mat.cols();

	// allocate PyArray
	#ifdef EIGEN_DEFAULT_TO_ROW_MAJOR
	PyObject* array = PyArray_New(&PyArray_Type, 2, dims, NPY_DOUBLE, 0, 0, sizeof(double), NPY_C_CONTIGUOUS, 0);
	#else
	PyObject* array = PyArray_New(&PyArray_Type, 2, dims, NPY_DOUBLE, 0, 0, sizeof(double), NPY_F_CONTIGUOUS, 0);
	#endif

	// copy data
	const double* data = mat.data();
	double* dataCopy = reinterpret_cast<double*>(PyArray_DATA(array));

	for(int i = 0; i < mat.size(); ++i)
		dataCopy[i] = data[i];

	return array;
}



// TODO: fix mess with 64 bit types
PyObject* PyArray_FromMatrixXi(const MatrixXi& mat) {
	// matrix dimensionality
	npy_intp dims[2];
	dims[0] = mat.rows();
	dims[1] = mat.cols();

	// allocate PyArray
	#ifdef EIGEN_DEFAULT_TO_ROW_MAJOR
	PyObject* array = PyArray_New(&PyArray_Type, 2, dims, NPY_INT64, 0, 0, sizeof(int64_t), NPY_C_CONTIGUOUS, 0);
	#else
	PyObject* array = PyArray_New(&PyArray_Type, 2, dims, NPY_INT64, 0, 0, sizeof(int64_t), NPY_F_CONTIGUOUS, 0);
	#endif

	// copy data
	Matrix<int64_t, Dynamic, Dynamic> tmp = mat.cast<int64_t>();
	const int64_t* data = tmp.data();
	int64_t* dataCopy = reinterpret_cast<int64_t*>(PyArray_DATA(array));

	for(int i = 0; i < mat.size(); ++i)
		dataCopy[i] = data[i];

	return array;
}



PyObject* PyArray_FromMatrixXb(const MatrixXb& mat) {
	// matrix dimensionality
	npy_intp dims[2];
	dims[0] = mat.rows();
	dims[1] = mat.cols();

	// allocate PyArray
	#ifdef EIGEN_DEFAULT_TO_ROW_MAJOR
	PyObject* array = PyArray_New(&PyArray_Type, 2, dims, NPY_BOOL, 0, 0, sizeof(bool), NPY_C_CONTIGUOUS, 0);
	#else
	PyObject* array = PyArray_New(&PyArray_Type, 2, dims, NPY_BOOL, 0, 0, sizeof(bool), NPY_F_CONTIGUOUS, 0);
	#endif

	// copy data
	const bool* data = mat.data();
	bool* dataCopy = reinterpret_cast<bool*>(PyArray_DATA(array));

	for(int i = 0; i < mat.size(); ++i)
		dataCopy[i] = data[i];

	return array;
}



MatrixXd PyArray_ToMatrixXd(PyObject* array) {
	if(PyArray_DESCR(array)->type != PyArray_DescrFromType(NPY_DOUBLE)->type)
		throw Exception("Can only handle arrays of double values.");

	if(PyArray_NDIM(array) == 1) {
		if(PyArray_FLAGS(array) & NPY_F_CONTIGUOUS)
			return Map<Matrix<double, Dynamic, Dynamic, ColMajor> >(
				reinterpret_cast<double*>(PyArray_DATA(array)),
				PyArray_DIM(array, 0), 1);

		else if(PyArray_FLAGS(array) & NPY_C_CONTIGUOUS)
			return Map<Matrix<double, Dynamic, Dynamic, RowMajor> >(
				reinterpret_cast<double*>(PyArray_DATA(array)),
				PyArray_DIM(array, 0), 1);

		else
			throw Exception("Data must be stored in contiguous memory.");

	} else if(PyArray_NDIM(array) == 2) {
		if(PyArray_FLAGS(array) & NPY_F_CONTIGUOUS)
			return Map<Matrix<double, Dynamic, Dynamic, ColMajor> >(
				reinterpret_cast<double*>(PyArray_DATA(array)),
				PyArray_DIM(array, 0),
				PyArray_DIM(array, 1));

		else if(PyArray_FLAGS(array) & NPY_C_CONTIGUOUS)
			return Map<Matrix<double, Dynamic, Dynamic, RowMajor> >(
				reinterpret_cast<double*>(PyArray_DATA(array)),
				PyArray_DIM(array, 0),
				PyArray_DIM(array, 1));

		else
			throw Exception("Data must be stored in contiguous memory.");

	} else {
		throw Exception("Can only handle one- and two-dimensional arrays.");
	}
}



// TODO: fix mess with 64 bit types
MatrixXi PyArray_ToMatrixXi(PyObject* array) {
	if(PyArray_DESCR(array)->type != PyArray_DescrFromType(NPY_INT64)->type)
		throw Exception("Can only handle arrays of integer values.");

	if(PyArray_NDIM(array) == 1) {
		if(PyArray_FLAGS(array) & NPY_F_CONTIGUOUS)
			return Map<Matrix<int64_t, Dynamic, Dynamic, ColMajor> >(
				reinterpret_cast<int64_t*>(PyArray_DATA(array)),
				PyArray_DIM(array, 0), 1).cast<int>();

		else if(PyArray_FLAGS(array) & NPY_C_CONTIGUOUS)
			return Map<Matrix<int64_t, Dynamic, Dynamic, RowMajor> >(
				reinterpret_cast<int64_t*>(PyArray_DATA(array)),
				PyArray_DIM(array, 0), 1).cast<int>();

		else
			throw Exception("Data must be stored in contiguous memory.");

	} else if(PyArray_NDIM(array) == 2) {
		if(PyArray_FLAGS(array) & NPY_F_CONTIGUOUS)
			return Map<Matrix<int64_t, Dynamic, Dynamic, ColMajor> >(
				reinterpret_cast<int64_t*>(PyArray_DATA(array)),
				PyArray_DIM(array, 0),
				PyArray_DIM(array, 1)).cast<int>();

		else if(PyArray_FLAGS(array) & NPY_C_CONTIGUOUS)
			return Map<Matrix<int64_t, Dynamic, Dynamic, RowMajor> >(
				reinterpret_cast<int64_t*>(PyArray_DATA(array)),
				PyArray_DIM(array, 0),
				PyArray_DIM(array, 1)).cast<int>();

		else
			throw Exception("Data must be stored in contiguous memory.");

	} else {
		throw Exception("Can only handle one- and two-dimensional arrays.");
	}
}



MatrixXb PyArray_ToMatrixXb(PyObject* array) {
	if(PyArray_DESCR(array)->type != PyArray_DescrFromType(NPY_BOOL)->type)
		throw Exception("Can only handle arrays of Boolean values.");

	if(PyArray_NDIM(array) == 1) {
		if(PyArray_FLAGS(array) & NPY_F_CONTIGUOUS)
			return Map<Matrix<bool, Dynamic, Dynamic, ColMajor> >(
				reinterpret_cast<bool*>(PyArray_DATA(array)),
				PyArray_DIM(array, 0), 1);

		else if(PyArray_FLAGS(array) & NPY_C_CONTIGUOUS)
			return Map<Matrix<bool, Dynamic, Dynamic, ColMajor> >(
				reinterpret_cast<bool*>(PyArray_DATA(array)),
				PyArray_DIM(array, 0), 1);

		else
			throw Exception("Data must be stored in contiguous memory.");

	} else if(PyArray_NDIM(array) == 2) {
		if(PyArray_FLAGS(array) & NPY_F_CONTIGUOUS)
			return Map<Matrix<bool, Dynamic, Dynamic, ColMajor> >(
				reinterpret_cast<bool*>(PyArray_DATA(array)),
				PyArray_DIM(array, 0),
				PyArray_DIM(array, 1));

		else if(PyArray_FLAGS(array) & NPY_C_CONTIGUOUS)
			return Map<Matrix<bool, Dynamic, Dynamic, RowMajor> >(
				reinterpret_cast<bool*>(PyArray_DATA(array)),
				PyArray_DIM(array, 0),
				PyArray_DIM(array, 1));

		else
			throw Exception("Data must be stored in contiguous memory.");

	} else {
		throw Exception("Can only handle one- and two-dimensional arrays.");
	}
}



vector<ArrayXXd> PyArray_ToArraysXXd(PyObject* array) {
	if(PyArray_DESCR(array)->type != PyArray_DescrFromType(NPY_DOUBLE)->type)
		throw Exception("Can only handle arrays of double values.");

	if(PyArray_NDIM(array) == 3) {
		vector<ArrayXXd> channels;

		if(PyArray_FLAGS(array) & NPY_F_CONTIGUOUS)
			for(int m = 0; m < PyArray_DIM(array, 2); ++m)
				channels.push_back(Matrix<double, Dynamic, Dynamic, ColMajor>(
					PyArray_DIM(array, 0),
					PyArray_DIM(array, 1)));

		else if(PyArray_FLAGS(array) & NPY_C_CONTIGUOUS)
			for(int m = 0; m < PyArray_DIM(array, 2); ++m)
				channels.push_back(Matrix<double, Dynamic, Dynamic, RowMajor>(
					PyArray_DIM(array, 0),
					PyArray_DIM(array, 1)));

		else
			throw Exception("Data must be stored in contiguous memory.");

		double* data = reinterpret_cast<double*>(PyArray_DATA(array));

		for(int m = 0; m < channels.size(); ++m)
			for(int i = 0; i < channels[m].size(); ++i)
				channels[m].data()[i] = data[m * channels[m].size() + i];

		return channels;
	} else {
		throw Exception("Can only handle three-dimensional arrays.");
	}
}



vector<ArrayXXb> PyArray_ToArraysXXb(PyObject* array) {
	if(PyArray_DESCR(array)->type != PyArray_DescrFromType(NPY_BOOL)->type)
		throw Exception("Can only handle arrays of bool values.");

	if(PyArray_NDIM(array) == 3) {
		vector<ArrayXXb> channels;

		if(PyArray_FLAGS(array) & NPY_F_CONTIGUOUS)
			for(int m = 0; m < PyArray_DIM(array, 2); ++m)
				channels.push_back(Matrix<bool, Dynamic, Dynamic, ColMajor>(
					PyArray_DIM(array, 0),
					PyArray_DIM(array, 1)));

		else if(PyArray_FLAGS(array) & NPY_C_CONTIGUOUS)
			for(int m = 0; m < PyArray_DIM(array, 2); ++m)
				channels.push_back(Matrix<bool, Dynamic, Dynamic, RowMajor>(
					PyArray_DIM(array, 0),
					PyArray_DIM(array, 1)));

		else
			throw Exception("Data must be stored in contiguous memory.");

		bool* data = reinterpret_cast<bool*>(PyArray_DATA(array));

		for(int m = 0; m < channels.size(); ++m)
			for(int i = 0; i < channels[m].size(); ++i)
				channels[m].data()[i] = data[m * channels[m].size() + i];

		return channels;
	} else {
		throw Exception("Can only handle three-dimensional arrays.");
	}
}



PyObject* PyArray_FromArraysXXd(const vector<ArrayXXd>& channels) {
	// matrix dimensionality
	npy_intp dims[3];
	dims[0] = channels[0].rows();
	dims[1] = channels[0].cols();
	dims[2] = channels.size();

	// allocate PyArray
	#ifdef EIGEN_DEFAULT_TO_ROW_MAJOR
	PyObject* array = PyArray_New(&PyArray_Type, 3, dims, NPY_DOUBLE, 0, 0, sizeof(double), NPY_C_CONTIGUOUS, 0);
	#else
	PyObject* array = PyArray_New(&PyArray_Type, 3, dims, NPY_DOUBLE, 0, 0, sizeof(double), NPY_F_CONTIGUOUS, 0);
	#endif

	// copy data
	double* dataCopy = reinterpret_cast<double*>(PyArray_DATA(array));

	for(int m = 0; m < channels.size(); ++m) {
		const double* data = channels[m].data();

		for(int i = 0; i < channels[m].size(); ++i)
			dataCopy[m * channels[m].size() + i] = data[i];
	}

	return array;
}



Tuples PyList_AsTuples(PyObject* list) {
	if(!PyList_Check(list))
		throw Exception("Indices should be given in a list.");

	Tuples tuples;

	// convert list of tuples
	for(int i = 0; i < PyList_Size(list); ++i) {
		PyObject* tuple = PyList_GetItem(list, i);

		if(!PyTuple_Check(tuple) || PyTuple_Size(tuple) != 2)
			throw Exception("Indices should be stored in a list of 2-tuples.");

		int m, n;

		if(!PyArg_ParseTuple(tuple, "ii", &m, &n))
			throw Exception("Indices should be integers.");

		tuples.push_back(make_pair(m, n));
	}

	return tuples;
}



PyObject* PyList_FromTuples(const Tuples& tuples) {
	PyObject* list = PyList_New(tuples.size());

	for(int i = 0; i < tuples.size(); ++i)
		PyList_SetItem(list, i,
			Py_BuildValue("(ii)", tuples[i].first, tuples[i].second));

	return list;
}



Regularizer PyObject_ToRegularizer(PyObject* regularizer) {
	if(PyFloat_Check(regularizer))
		return Regularizer(PyFloat_AsDouble(regularizer));

	if(PyInt_Check(regularizer))
		return Regularizer(static_cast<double>(PyInt_AsLong(regularizer)));

	if(PyDict_Check(regularizer)) {
		PyObject* r_strength = PyDict_GetItemString(regularizer, "strength");
		PyObject* r_transform = PyDict_GetItemString(regularizer, "transform");
		PyObject* r_norm = PyDict_GetItemString(regularizer, "norm");

		if(r_transform == Py_None)
			r_transform = 0;

		Regularizer::Norm norm = Regularizer::L2;

		if(r_norm) {
			if(PyString_Size(r_norm) != 2)
				throw Exception("Regularizer norm should be 'L1' or 'L2'.");

			switch(PyString_AsString(r_norm)[1]) {
				default:
					throw Exception("Regularizer norm should be 'L1' or 'L2'.");

				case '1':
					norm = Regularizer::L1;
					break;

				case '2':
					norm = Regularizer::L2;
					break;
			}
		}

		double strength = r_transform ? 1. : 0.;

		if(r_strength) {
			if(PyInt_Check(r_strength)) {
				strength = static_cast<double>(PyInt_AsLong(r_strength));
			} else {
				if(!PyFloat_Check(r_strength))
					throw Exception("Regularizer strength should be of type `float`.");
				strength = PyFloat_AsDouble(r_strength);
			}
		}

		if(r_transform) {
			PyObject* matrix = PyArray_FROM_OTF(r_transform, NPY_DOUBLE, NPY_IN_ARRAY);

			if(!matrix)
				throw Exception("Regularizer transform should be of type `ndarray`.");

			return Regularizer(PyArray_ToMatrixXd(matrix), norm, strength);
		} else {
			return Regularizer(strength, norm);
		}
	}

	PyObject* matrix = PyArray_FROM_OTF(regularizer, NPY_DOUBLE, NPY_IN_ARRAY);

	if(matrix)
		return Regularizer(PyArray_ToMatrixXd(matrix));

	throw Exception("Regularizer should be of type `dict`, `float` or `ndarray`.");
}
