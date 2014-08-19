#include "MEX/Output.h"

using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::ColMajor;

#include <iostream>

MEX::Output::Output() : mSize(0), mData(NULL) {
}

MEX::Output::Output(int size, mxArray** data) : mSize(size), mData(data) {
    // ToDo: Check if we really should do this
    memset(mData, 0, mSize * sizeof(mxArray*));
}

int MEX::Output::size() const {
    return mSize;
}

bool MEX::Output::has(int index) const {
    return index >= 0 && index < mSize;
}

MEX::Output::Setter MEX::Output::operator[](int index) const {
    if(!has(index)) {
        mexErrMsgIdAndTxt("MEX:Output:missingArgument", "Not enough output argument required! Could not access argument #%d.", index + 1);
    }

    // Avoid memory leaks and garbage collection by deleting previous allocated data.
    if(mData[index] != NULL) {
        mxDestroyArray(mData[index]);
        mData[index] = NULL;
    }

    return Setter(mData + index);
}



MEX::Output::Setter::Setter(mxArray** data) : mData(data) {
}

MEX::Output::Setter& MEX::Output::Setter::operator=(const MatrixXd& output) {
    (*mData) = mxCreateDoubleMatrix(output.rows(), output.cols(), mxREAL);
    Map<MatrixXd,ColMajor> data_wrapper(mxGetPr(*mData), output.rows(), output.cols());
    data_wrapper = output;
    return *this;
}

// MEX::Output::Setter& MEX::Output::Setter::operator=(const MatrixXb& output) {
//     (*mData) = mxCreateLogicalMatrix(output.rows(), output.cols());
//     Map<MatrixXb,ColMajor> data_wrapper(mxGetPr(*mData), output.rows(), output.cols());
//     data_wrapper = output;
//     return *this;
// }

MEX::Output::Setter& MEX::Output::Setter::operator=(const bool& b) {
    (*mData) = mxCreateLogicalScalar(b);
    return *this;
}

MEX::Output::Setter& MEX::Output::Setter::operator=(const int& i) {
    (*mData) = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
    (*(int*)mxGetData(*mData)) = i;
    return *this;
}

MEX::Output::Setter& MEX::Output::Setter::operator=(const double& d) {
    (*mData) = mxCreateDoubleScalar(d);
    return *this;
}

MEX::Output::Setter& MEX::Output::Setter::operator=(const std::string& s) {
    (*mData) = mxCreateString(s.c_str());
    return *this;
}

MEX::Output::Setter& MEX::Output::Setter::operator=(const mxArray* a) {
	(*mData) = mxDuplicateArray(a);
	return *this;
}

MEX::Output::Setter& MEX::Output::Setter::operator=(const MEX::Output::Setter& s) {
    (*mData) = mxDuplicateArray(*s.mData);
    return *this;
}



