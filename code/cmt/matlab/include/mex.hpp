/*
Copyright (c) 2012, Oliver Woodford
Copyright (c) 2014, Florian Franzen
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the distribution

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef __MEX_HPP__
#define __MEX_HPP__

#include "mex.h"
#include <stdint.h>
#include <string>
#include <cstring>
#include <typeinfo>
#include <iostream>
#include <sstream>
#include <exception>

#include "Eigen/Core"
using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::ColMajor;


#define MEX_CLASS_HANDLE_SIGNATURE 0x434d544d

template<class BaseClass> class MEXClassHandle {
public:

    static mxArray* wrap(BaseClass* ptr) {
        mexLock();
        mxArray *out = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
        *((uint64_t *)mxGetData(out)) = reinterpret_cast<uint64_t>(new MEXClassHandle<BaseClass>(ptr));
        return out;
    }

    static BaseClass* unwrap(const mxArray *in) {
        return fromMxArray(in)->getPointer();
    }

    static void free(const mxArray *in) {
        delete fromMxArray(in);
        mexUnlock();
    }

private:
    MEXClassHandle(BaseClass* pointer) : mPointer(pointer), mName(typeid(BaseClass).name()), mSignature(MEX_CLASS_HANDLE_SIGNATURE) {}

    ~MEXClassHandle() {
        mSignature = 0;
        delete mPointer;
    }

    bool isValid() {
        return ((mSignature == MEX_CLASS_HANDLE_SIGNATURE) &&
                !strcmp(mName.c_str(), typeid(BaseClass).name()));
    }

    BaseClass* getPointer() {
        return mPointer;
    }

    static MEXClassHandle* fromMxArray(const mxArray *in) {
        if (mxGetNumberOfElements(in) != 1 || !mxIsUint64(in) || mxIsComplex(in))
            mexErrMsgIdAndTxt("MEXClassHandle:notAHandle", "Handle must be a real uint64 scalar.");

        MEXClassHandle<BaseClass> *handle = reinterpret_cast<MEXClassHandle<BaseClass> *>(*((uint64_t*) mxGetData(in)));

        if (!handle->isValid())
            mexErrMsgIdAndTxt("MEXClassHandle:invalidHandle", "Handle is not valid.");

        return handle;
    }

    uint32_t    mSignature;
    std::string mName;
    BaseClass*  mPointer;
};

class MEXInput {
public:
    MEXInput(int size, const mxArray** data) : mSize(size), mData(data) {}

    int size() const {
        return mSize;
    }

    bool has(int i) const {
        return i >= 0 && i < mSize;
    }

    enum Type { // ToDo: Decide if you want to add vectors
        Unknown,
        BoolScalar,
        IntScalar,
        FloatScalar,
        BoolMatrix,
        IntMatrix,
        FloatMatrix,
        String,
        Struct,
        Cell
    };

    class Converter {
    public:
        Converter(const mxArray* data, int index) : mData(data), mIndex(index) {}

        bool isEmpty() {
            return mxIsEmpty(mData);
        }

        MEXInput::Type getType() {
            switch (mxGetClassID(mData)) {
                case mxCHAR_CLASS:
                    return MEXInput::String;
                case mxSTRUCT_CLASS:
                    return MEXInput::Struct;
                case mxCELL_CLASS:
                    return MEXInput::Cell;
                case mxLOGICAL_CLASS:
                    return (mxGetNumberOfElements(mData) == 1) ? MEXInput::BoolScalar : MEXInput::BoolMatrix;
                case mxINT8_CLASS:
                case mxUINT8_CLASS:
                case mxINT16_CLASS:
                case mxUINT16_CLASS:
                case mxINT32_CLASS:
                case mxUINT32_CLASS:
                case mxINT64_CLASS:
                case mxUINT64_CLASS:
                    return (mxGetNumberOfElements(mData) == 1) ? MEXInput::IntScalar : MEXInput::IntMatrix;
                case mxSINGLE_CLASS:
                case mxDOUBLE_CLASS:
                    return (mxGetNumberOfElements(mData) == 1) ? MEXInput::FloatScalar : MEXInput::FloatMatrix;
                case mxUNKNOWN_CLASS:
                    mexErrMsgTxt("Unknown class.");
                default:
                    mexErrMsgTxt("Unidentified class.");
            }
            return MEXInput::Unknown;
        }

        bool isType(MEXInput::Type t) {
            return (getType() == t);
        }

        template<class BaseClass> BaseClass* unwrap() {
            return MEXClassHandle<BaseClass>::unwrap(mData);
        }

        operator MatrixXd () {
            if(!isType(MEXInput::FloatMatrix)){
                mexErrMsgIdAndTxt("MEXInput:typeMismatch", "Argument #%d should be a single or double matrix.", mIndex + 1);
            }

            if(mxIsSparse(mData)) {
                mexErrMsgIdAndTxt("MEXInput:sparseMatrixNotSupported", "Sparse matrix are not supported (argument #%d).", mIndex + 1); // ToDo: Add sparse matrix support.
            }

            if(mxGetNumberOfDimensions(mData) > 2) {
                mexErrMsgIdAndTxt("MEXInput:moreThenTwoDimensions", "3D matrix are not supported (argument #%d).", mIndex + 1); // ToDo: Add 3d matrix support.
            }

            return Map<MatrixXd,ColMajor>(mxGetPr(mData), mxGetM(mData), mxGetN(mData));
         }

        operator VectorXd () {
            if(!isType(MEXInput::FloatMatrix) || mxGetN(mData) != 1) {
                mexErrMsgIdAndTxt("MEXInput:typeMismatch", "Argument #%d should be a single or double column vector.", mIndex + 1);
            }

            if(mxIsSparse(mData)) {
                mexErrMsgIdAndTxt("MEXInput:sparseMatrixNotSupported", "Sparse matrix are not supported (argument #%d).", mIndex + 1); // ToDo: Add sparse matrix support.
            }

            if(mxGetNumberOfDimensions(mData) > 2) {
                mexErrMsgIdAndTxt("MEXInput:moreThenTwoDimensions", "3D matrix are not supported (argument #%d).", mIndex + 1); // ToDo: Add 3d matrix support.
            }

            return Map<VectorXd,ColMajor>(mxGetPr(mData), mxGetM(mData));

        }

        operator double () {
            if(!isType(MEXInput::FloatScalar)){
                mexErrMsgIdAndTxt("MEXInput:typeMismatch", "Argument #%d should be a single or double scalar.", mIndex + 1);
            }

            return mxGetScalar(mData);
        }

        operator int () {
            if(!isType(MEXInput::IntScalar)){
                mexErrMsgIdAndTxt("MEXInput:typeMismatch", "Argument #%d should be a integer scalar.", mIndex + 1);
            }

            return static_cast<int>(round(mxGetScalar(mData)));
        }

        operator bool () {
            if(!isType(MEXInput::BoolScalar)){
                mexErrMsgTxt("This should be a boolean scalar!");
            }

            return mxGetScalar(mData) != 0;
        }

    private:
        const mxArray* mData;
        int mIndex; // Only needed for error messages
    };

    MEXInput::Converter operator[](int i) const {
        if(!has(i)) {
            mexErrMsgIdAndTxt("MEXInput:missingArgument", "Not enough argument supplied! Could not access argument #%d.", i + 1);
        }

        return Converter(mData[i], i);
    }

private:
    int mSize;
    const mxArray** mData;
};


class MEXOutput {

public:
    MEXOutput(int size, mxArray** data) : mSize(size), mData(data) {}

    int size() {
        return mSize;
    }

    int has(int index) {
        return index >= 0 && index < mSize;
    }

    class Converter {
        public:
            Converter(mxArray** data) : mData(data) {}

            Converter& operator=(const MatrixXd& output) {
                (*mData) = mxCreateDoubleMatrix(output.rows(), output.cols(), mxREAL);
                Map<MatrixXd,ColMajor> data_wrapper(mxGetPr(*mData), output.rows(), output.cols());
                data_wrapper = output;
                return *this;
            }

            // Converter& operator=(const MatrixXb& output) {
            //     (*mData) = mxCreateLogicalMatrix(output.rows(), output.cols());
            //     Map<MatrixXb,ColMajor> data_wrapper(mxGetPr(*mData), output.rows(), output.cols());
            //     data_wrapper = output;
            //     return *this;
            // }

            Converter& operator=(const bool& b) {
                (*mData) = mxCreateLogicalScalar(b);
                return *this;
            }

            Converter& operator=(const int& i) {
                const mwSize dims = 1;
                (*mData) = mxCreateNumericArray(1, &dims, mxINT32_CLASS, mxREAL);
                (*mxGetPr(*mData)) = i;
                return *this;
            }

            Converter& operator=(const double& d) {
                std::cout << "Double: " << d << std::endl;
                (*mData) = mxCreateDoubleScalar(d);
                return *this;
            }

            Converter& operator=(const std::string& s) {
                (*mData) = mxCreateString(s.c_str());
                return *this;
            }

        private:
            mxArray** mData;
    };

    MEXOutput::Converter operator[](int index) {
        if(!has(index)) {
            mexErrMsgIdAndTxt("MEXOutput:missingArgument", "Not enough output argument required! Could not access argument #%d.", index + 1);
        }

        return Converter(mData + index);
    }

private:
    int mSize;
    mxArray** mData;
};

// streambuffer that writes everything to mexPrintf. Useful to reroute cout to matlab.
class mexstreambuf : public std::streambuf {
protected:
    std::streamsize xsputn(const char *s, std::streamsize n) {
        mexPrintf("%.*s",n,s);
        return n;
    }

    int overflow(int c = EOF) {
        if (c != EOF) {
            mexPrintf("%.1s",&c);
        }
        return 1;
    }

};

/**
 * Mex wrapper function, that takes care of object creation, storage and deletion and passes everything else onto parser function.
 *
 * @param creator function that parses matlab data to call class constructor
 * @param parser function that parses matlab data and forwards the appropriate function
 */
template<class BaseClass> inline void mexWrapper(BaseClass* (*creator)(MEXInput), bool (*parser)(BaseClass*, std::string, MEXOutput, MEXInput), int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Redirect cout
    mexstreambuf mexoutbuf;
    std::streambuf* coutbuf = std::cout.rdbuf(&mexoutbuf);

    // Get the command string
    char cmd[64];
    if (nrhs < 1 || mxGetString(prhs[0], cmd, sizeof(cmd)))
        mexErrMsgIdAndTxt("mexWrapper:invalidCommandstring", "First input should be a command string less than 64 characters long.");

    std::cout << "+++ DEBUG: Call to '" << cmd << "' with " << nrhs << " inputs and " << nlhs << " ouput value(s). +++" << std::endl;

    // New
    if (!strcmp("new", cmd)) {
        // Check parameters
        if (nlhs != 1)
            mexErrMsgIdAndTxt("mexWrapper:invalidOutput", "New: One output expected.");

        try {
            BaseClass* ptr = creator(MEXInput(nrhs - 1, prhs + 1));
            // Return a handle to a new C++ instance
            plhs[0] = MEXClassHandle<BaseClass>::wrap(ptr);
            return;
        } catch (std::exception& e) {
            mexErrMsgIdAndTxt("mexWrapper:constructor:exceptionCaught", "Exception in constructor: \n\t%s", e.what());
        }
    }

    // Check there is a second input, which should be the class instance handle
    if (nrhs < 2)
        mexErrMsgIdAndTxt("mexWrapper:missingClassHandle", "Second input should be a class instance handle.");

    // Delete
    if (!strcmp("delete", cmd)) {
        // Destroy the C++ object
        MEXClassHandle<BaseClass>::free(prhs[1]);
        // Warn if other commands were ignored
        if (nlhs != 0 || nrhs != 2)
            mexWarnMsgIdAndTxt("mexWrapper:ignoredArgurments", "Delete: Unexpected arguments ignored.");
        return;
    }

    // Get the class instance pointer from the second input
    BaseClass* instance = MEXClassHandle<BaseClass>::unwrap(prhs[1]);

    try {
        if(!parser(instance, cmd, MEXOutput(nlhs, plhs), MEXInput(nrhs - 2, prhs + 2)))
            mexErrMsgIdAndTxt("mexParse:unknownCommand", "Command '%s' not recognized!", cmd);
    } catch (std::exception& e) {
        mexErrMsgIdAndTxt("mexWrapper:methodAndPropertyParser:exceptionCaught", "Exception in method and property parser: \n\t%s", e.what());
    }

    // Reset cout
    std::cout.rdbuf(coutbuf);
}

#endif // __MEX_HPP__
