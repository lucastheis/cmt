#include "MEX/Input.h"

using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::VectorXd;
using Eigen::ArrayXXd;
using Eigen::ArrayXXi;
using Eigen::ColMajor;

using std::vector;

#include "MEX/Function.h"

MEX::Input::Input(int size, const mxArray** data) : mSize(size), mData(data) {
}

int MEX::Input::size() const {
    return mSize;
}

bool MEX::Input::has(int i) const {
    return i >= 0 && i < mSize;
}

MEX::Input::Getter::Getter(const mxArray* data, int index) : mData(data), mIndex(index) {
}

bool MEX::Input::Getter::isEmpty() {
    return mxIsEmpty(mData);
}

MEX::Type::Type MEX::Input::Getter::getType() {
    switch (mxGetClassID(mData)) {
        case mxCHAR_CLASS:
            return MEX::Type::String;
        case mxSTRUCT_CLASS:
            return MEX::Type::Struct;
        case mxCELL_CLASS:
            return MEX::Type::Cell;
        case mxLOGICAL_CLASS:
            return (mxGetNumberOfElements(mData) == 1) ? MEX::Type::BoolScalar : MEX::Type::BoolMatrix;
        case mxINT8_CLASS:
        case mxUINT8_CLASS:
        case mxINT16_CLASS:
        case mxUINT16_CLASS:
        case mxINT32_CLASS:
        case mxUINT32_CLASS:
        case mxINT64_CLASS:
        case mxUINT64_CLASS:
            return (mxGetNumberOfElements(mData) == 1) ? MEX::Type::IntScalar : MEX::Type::IntMatrix;
        case mxSINGLE_CLASS:
        case mxDOUBLE_CLASS:
            return (mxGetNumberOfElements(mData) == 1) ? MEX::Type::FloatScalar : MEX::Type::FloatMatrix;
        case mxFUNCTION_CLASS:
            return MEX::Type::Function;
        case mxUNKNOWN_CLASS:
            mexErrMsgTxt("Unknown class.");
        default:
            mexErrMsgTxt("Unidentified class.");
    }
    return MEX::Type::Unknown;
}

bool MEX::Input::Getter::isType(Type::Type t) {
    return (getType() == t);
}


bool MEX::Input::Getter::isClass(std::string name) {
    return mxIsClass(mData, name.c_str());
}


std::string MEX::Input::Getter::getClass() {
    return mxGetClassName(mData);
}


MEX::Input::Getter MEX::Input::Getter::getObjectProperty(std::string name) {
    mxArray* property = mxGetProperty(mData, 0, name.c_str());

    if(!property){
        mexErrMsgIdAndTxt("MEX:Input:unknownProperty", "Argument #%d has no property named '%s'.", mIndex + 1, name.c_str());
    }

    return Getter(property, mIndex);
}


MEX::Input::Getter::operator MatrixXd () {
    if(!isType(Type::FloatMatrix)){
        mexErrMsgIdAndTxt("MEX:Input:typeMismatch", "Argument #%d should be a single or double matrix.", mIndex + 1);
    }

    if(mxIsSparse(mData)) {
        mexErrMsgIdAndTxt("MEX:Input:sparseMatrixNotSupported", "Sparse matrix are not supported (argument #%d).", mIndex + 1); // ToDo: Add sparse matrix support.
    }

    if(mxGetNumberOfDimensions(mData) > 2) {
        mexErrMsgIdAndTxt("MEX:Input:moreThenTwoDimensions", "3D matrix are not supported (argument #%d).", mIndex + 1); // ToDo: Add 3d matrix support.
    }

    return Map<MatrixXd,ColMajor>(mxGetPr(mData), mxGetM(mData), mxGetN(mData));
 }

MEX::Input::Getter::operator MatrixXi () {
    if(mxGetClassID(mData) != mxINT32_CLASS) {
        mexErrMsgIdAndTxt("MEX:Input:typeMismatch", "Argument #%d should be a 32-bit integer matrix.", mIndex + 1);
    }

    if(mxIsSparse(mData)) {
        mexErrMsgIdAndTxt("MEX:Input:sparseMatrixNotSupported", "Sparse matrix are not supported (argument #%d).", mIndex + 1); // ToDo: Add sparse matrix support.
    }

    if(mxGetNumberOfDimensions(mData) > 2) {
        mexErrMsgIdAndTxt("MEX:Input:moreThenTwoDimensions", "3D matrix are not supported (argument #%d).", mIndex + 1); // ToDo: Add 3d matrix support.
    }

    return Map<ArrayXXi,ColMajor>((int*) mxGetData(mData), mxGetM(mData), mxGetN(mData));
}

MEX::Input::Getter::operator VectorXd () {
    if(!isType(Type::FloatMatrix) || mxGetN(mData) != 1) {
        mexErrMsgIdAndTxt("MEX:Input:typeMismatch", "Argument #%d should be a single or double column vector.", mIndex + 1);
    }

    if(mxIsSparse(mData)) {
        mexErrMsgIdAndTxt("MEX:Input:sparseMatrixNotSupported", "Sparse matrix are not supported (argument #%d).", mIndex + 1); // ToDo: Add sparse matrix support.
    }

    if(mxGetNumberOfDimensions(mData) > 2) {
        mexErrMsgIdAndTxt("MEX:Input:moreThenTwoDimensions", "3D matrix are not supported (argument #%d).", mIndex + 1); // ToDo: Add 3d matrix support.
    }

    return Map<VectorXd,ColMajor>(mxGetPr(mData), mxGetM(mData));

}

MEX::Input::Getter::operator ArrayXXd () {
    if(!isType(Type::FloatMatrix)) {
        mexErrMsgIdAndTxt("MEX:Input:typeMismatch", "Argument #%d should be a single or double matrix.", mIndex + 1);
    }

    if(mxIsSparse(mData)) {
        mexErrMsgIdAndTxt("MEX:Input:sparseMatrixNotSupported", "Sparse matrix are not supported (argument #%d).", mIndex + 1); // ToDo: Add sparse matrix support.
    }

    if(mxGetNumberOfDimensions(mData) > 2) {
        mexErrMsgIdAndTxt("MEX:Input:moreThenTwoDimensions", "3D matrix are not supported (argument #%d).", mIndex + 1); // ToDo: Add 3d matrix support.
    }

    return Map<ArrayXXd,ColMajor>(mxGetPr(mData), mxGetM(mData), mxGetN(mData));
}

MEX::Input::Getter::operator ArrayXXi () {
    if(mxGetClassID(mData) != mxINT32_CLASS) {
        mexErrMsgIdAndTxt("MEX:Input:typeMismatch", "Argument #%d should be a 32-bit integer matrix.", mIndex + 1);
    }

    if(mxIsSparse(mData)) {
        mexErrMsgIdAndTxt("MEX:Input:sparseMatrixNotSupported", "Sparse matrix are not supported (argument #%d).", mIndex + 1); // ToDo: Add sparse matrix support.
    }

    if(mxGetNumberOfDimensions(mData) > 2) {
        mexErrMsgIdAndTxt("MEX:Input:moreThenTwoDimensions", "3D matrix are not supported (argument #%d).", mIndex + 1); // ToDo: Add 3d matrix support.
    }

    return Map<ArrayXXi,ColMajor>((int*) mxGetData(mData), mxGetM(mData), mxGetN(mData));
}

// ToDo: Fix API so this is not needed.
MEX::Input::Getter::operator Eigen::Array<int, 1, Eigen::Dynamic> () {
    if(mxGetClassID(mData) != mxINT32_CLASS || mxGetM(mData) > 1) {
        mexErrMsgIdAndTxt("MEX:Input:typeMismatch", "Argument #%d should be a 32-bit integer vector.", mIndex + 1);
    }

    if(mxIsSparse(mData)) {
        mexErrMsgIdAndTxt("MEX:Input:sparseMatrixNotSupported", "Sparse matrix are not supported (argument #%d).", mIndex + 1); // ToDo: Add sparse matrix support.
    }

    if(mxGetNumberOfDimensions(mData) > 2) {
        mexErrMsgIdAndTxt("MEX:Input:moreThenTwoDimensions", "3D matrix are not supported (argument #%d).", mIndex + 1); // ToDo: Add 3d matrix support.
    }

    return Map<Eigen::Array<int, 1, Eigen::Dynamic>,ColMajor>((int*) mxGetData(mData), 1, mxGetN(mData));
}


MEX::Input::Getter::operator vector<MatrixXd> () {
    if(!isType(Type::Cell)) {
        mexErrMsgIdAndTxt("MEX:Input:typeMismatch", "Argument #%d should be a cell of single or double matrices.", mIndex + 1);
    }

    int num_cells = mxGetNumberOfElements(mData);

    vector<MatrixXd> ret(num_cells);

    for(int index = 0; index < num_cells; index++) {
        ret[index] = Getter(mxGetCell(mData, index), mIndex);
    }

    return ret;
}

MEX::Input::Getter::operator double () {
    if(!isType(Type::FloatScalar)) {
        mexErrMsgIdAndTxt("MEX:Input:typeMismatch", "Argument #%d should be a single or double scalar.", mIndex + 1);
    }

    return mxGetScalar(mData);
}

MEX::Input::Getter::operator std::string () {
    if(!isType(Type::String)) {
        mexErrMsgIdAndTxt("MEX:Input:typeMismatch", "Argument #%d should be a string.", mIndex + 1);
    }

    char* buf = mxArrayToString(mData);
    std::string s(buf);
    mxFree(buf);
    return s;
}

MEX::Input::Getter::operator int () {
    if(isType(Type::FloatScalar)) {
        double int_part;
        if(modf(mxGetScalar(mData), &int_part) != 0.0) {
            // Maybe this should just be a warning...
            mexErrMsgIdAndTxt("MEX:Input:typeMismatch", "Argument #%d should be a integer scalar.", mIndex + 1);
        }

        return static_cast<int>(int_part);
    } else if(!isType(Type::IntScalar)) {
        mexErrMsgIdAndTxt("MEX:Input:typeMismatch", "Argument #%d should be a integer scalar.", mIndex + 1);
    }

    return static_cast<int>(mxGetScalar(mData));
}

MEX::Input::Getter::operator bool () {
    if(!isType(Type::BoolScalar)){
        mexErrMsgIdAndTxt("MEX:Input:typeMismatch", "Argument #%d should be a boolean scalar.", mIndex + 1);
    }

    return mxGetScalar(mData) != 0;
}

MEX::Input::Getter::operator MEX::Function () {
    if(!isType(Type::Function) && !isType(Type::String)){
        mexErrMsgIdAndTxt("MEX:Input:typeMismatch", "Argument #%d should be a function handle.", mIndex + 1);
    }

    return Function(mData);
}


MEX::Input::Getter MEX::Input::operator[](int i) const {
    if(!has(i)) {
        mexErrMsgIdAndTxt("MEX:Input:missingArgument", "Not enough argument supplied! Could not access argument #%d.", i + 1);
    }

    return Getter(mData[i], i);
}
