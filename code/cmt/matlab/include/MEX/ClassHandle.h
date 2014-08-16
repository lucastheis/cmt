#ifndef __MEX_CLASS_HANDLE_H__
#define __MEX_CLASS_HANDLE_H__

#include <typeinfo>
#include <string>

#include "mex.h"

#define MEX_CLASS_HANDLE_SIGNATURE 0x434d544d

namespace MEX {
    template<class BaseClass> class ClassHandle {
    public:

        static mxArray* wrap(BaseClass* ptr) {
            mexLock();
            mxArray *out = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
            *((uint64_t *)mxGetData(out)) = reinterpret_cast<uint64_t>(new ClassHandle<BaseClass>(ptr));
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
        ClassHandle(BaseClass* pointer) : mPointer(pointer), mName(typeid(BaseClass).name()), mSignature(MEX_CLASS_HANDLE_SIGNATURE) {}

        ~ClassHandle() {
            mSignature = 0;
            delete mPointer; // ToDo: This probably should not be here!
        }

        bool isValid() {
            return ((mSignature == MEX_CLASS_HANDLE_SIGNATURE) &&
                    !strcmp(mName.c_str(), typeid(BaseClass).name()));
        }

        BaseClass* getPointer() {
            return mPointer;
        }

        static ClassHandle* fromMxArray(const mxArray *in) {
            if (mxGetNumberOfElements(in) != 1 || !mxIsUint64(in) || mxIsComplex(in))
                mexErrMsgIdAndTxt("MEX:ClassHandle:notAHandle", "Handle must be a real uint64 scalar.");

            ClassHandle<BaseClass> *handle = reinterpret_cast<ClassHandle<BaseClass> *>(*((uint64_t*) mxGetData(in)));

            if (!handle->isValid())
                mexErrMsgIdAndTxt("MEX:ClassHandle:invalidHandle", "Handle is not valid.");

            return handle;
        }

        uint32_t    mSignature;
        std::string mName;
        BaseClass*  mPointer;
    };
}

#endif
