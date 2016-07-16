#include "MEX/Output.h"

using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::ArrayXXd;
using Eigen::ArrayXXi;
using Eigen::ColMajor;

using std::vector;

#include <iostream>

MEX::Output::Output() : mSize(0), mData(NULL) {
}

MEX::Output::Output(int size, mxArray** data) : mSize(size), mData(data) {
    // Zero out pointer array, so we can use !ptr to check for validity.
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

    return Setter(mData + index);
}



MEX::Output::Setter::Setter(mxArray** data) : mData(data) {
}

MEX::Output::Setter& MEX::Output::Setter::operator=(const MatrixXd& output) {
    this->clear();
    (*mData) = mxCreateDoubleMatrix(output.rows(), output.cols(), mxREAL);
    Map<MatrixXd,ColMajor> data_wrapper(mxGetPr(*mData), output.rows(), output.cols());
    data_wrapper = output;
    return *this;
}

MEX::Output::Setter& MEX::Output::Setter::operator=(const ArrayXXd& output) {
    this->clear();
    (*mData) = mxCreateDoubleMatrix(output.rows(), output.cols(), mxREAL);
    Map<ArrayXXd,ColMajor> data_wrapper(mxGetPr(*mData), output.rows(), output.cols());
    data_wrapper = output;
    return *this;
}

MEX::Output::Setter& MEX::Output::Setter::operator=(const ArrayXXi& output) {
    this->clear();
    (*mData) = mxCreateNumericMatrix(output.rows(), output.cols(), mxINT32_CLASS, mxREAL);
    Map<ArrayXXi,ColMajor> data_wrapper((int*) mxGetData(*mData), output.rows(), output.cols());
    data_wrapper = output;
    return *this;
}

// MEX::Output::Setter& MEX::Output::Setter::operator=(const MatrixXb& output) {
//     (*mData) = mxCreateLogicalMatrix(output.rows(), output.cols());
//     Map<MatrixXb,ColMajor> data_wrapper(mxGetPr(*mData), output.rows(), output.cols());
//     data_wrapper = output;
//     return *this;
// }

MEX::Output::Setter& MEX::Output::Setter::operator=(const vector<MatrixXd>& output) {
    this->clear();
    (*mData) = mxCreateCellMatrix(1, output.size());
    for(int i = 0; i < output.size(); i++){
        mxArray* data = mxCreateDoubleMatrix(output[i].rows(), output[i].cols(), mxREAL);
        Map<MatrixXd,ColMajor> data_wrapper(mxGetPr(data), output[i].rows(), output[i].cols());
        data_wrapper = output[i];
        mxSetCell(*mData, i, data);
    }
    return *this;
}

MEX::Output::Setter& MEX::Output::Setter::operator=(const bool& b) {
    this->clear();
    (*mData) = mxCreateLogicalScalar(b);
    return *this;
}

MEX::Output::Setter& MEX::Output::Setter::operator=(const int& i) {
    this->clear();
    (*mData) = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
    (*(int*)mxGetData(*mData)) = i;
    return *this;
}

MEX::Output::Setter& MEX::Output::Setter::operator=(const double& d) {
    this->clear();
    (*mData) = mxCreateDoubleScalar(d);
    return *this;
}

MEX::Output::Setter& MEX::Output::Setter::operator=(const std::string& s) {
    this->clear();
    (*mData) = mxCreateString(s.c_str());
    return *this;
}

MEX::Output::Setter& MEX::Output::Setter::operator=(const mxArray* a) {
    this->clear();
	(*mData) = mxDuplicateArray(a);
	return *this;
}

MEX::Output::Setter& MEX::Output::Setter::operator=(const MEX::Output::Setter& s) {
    this->clear();
    (*mData) = mxDuplicateArray(*s.mData);
    return *this;
}

void MEX::Output::Setter::clear() {
    // Avoid memory leaks and garbage collection by deleting previous allocated data.
    if(!(*mData)) {
        mxDestroyArray(*mData);
        (*mData) = NULL;
    }
}
