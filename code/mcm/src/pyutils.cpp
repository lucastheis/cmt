#include "pyutils.h"
#include "exception.h"
#include <cinttypes>

using std::int64_t;

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
		throw Exception("Can only handle one- or two-dimensional arrays.");
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
		throw Exception("Can only handle one- or two-dimensional arrays.");
	}
}
