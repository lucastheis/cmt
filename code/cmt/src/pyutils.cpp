#include "pyutils.h"
#include <inttypes.h>

#include "exception.h"
using CMT::Exception;

#include "Eigen/Core"
using Eigen::Map;
using Eigen::ColMajor;
using Eigen::RowMajor;

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
	const int64_t* data = mat.cast<int64_t>().eval().data();
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
