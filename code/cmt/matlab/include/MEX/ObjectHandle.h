#ifndef __MEX_CLASS_HANDLE_H__
#define __MEX_CLASS_HANDLE_H__

#include <typeinfo>
#include <string>
#include <stdint.h>

#include "mex.h"

#define MEX_OBJECT_HANDLE_SIGNATURE 0x434d544d

namespace MEX {
    template<class BaseClass> class ObjectHandle {
    public:

        static mxArray* share(BaseClass* ptr) {
            return toMxArray(ptr, false);
        }

        static mxArray* wrap(BaseClass* ptr) {
            mexLock();
            return toMxArray(ptr, true);
        }

        static BaseClass* unwrap(const mxArray *in) {
            return fromMxArray(in)->getPointer();
        }

        static void free(const mxArray *in) {
            ObjectHandle* obj = fromMxArray(in);
            bool owner = obj->mOwner;
            delete obj;
            if(owner) mexUnlock();
        }

    private:
        ObjectHandle(BaseClass* pointer, bool owner) : mPointer(pointer),
                                                       mName(typeid(BaseClass).name()),
                                                       mSignature(MEX_OBJECT_HANDLE_SIGNATURE),
                                                       mOwner(owner) {
        }

        ~ObjectHandle() {
            mSignature = 0;
            if(mOwner) delete mPointer;
        }

        bool isValid() {
            return ((mSignature == MEX_OBJECT_HANDLE_SIGNATURE) &&
                    !strcmp(mName.c_str(), typeid(BaseClass).name()));
        }

        BaseClass* getPointer() {
            return mPointer;
        }

        static mxArray* toMxArray(BaseClass* ptr, bool owner) {
            mxArray *out = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
            *((uint64_t *)mxGetData(out)) = reinterpret_cast<uint64_t>(new ObjectHandle<BaseClass>(ptr, owner));
            return out;
        }

        static ObjectHandle* fromMxArray(const mxArray *in) {
            if (mxGetNumberOfElements(in) != 1 || !mxIsUint64(in) || mxIsComplex(in))
                mexErrMsgIdAndTxt("MEX:ObjectHandle:notAHandle", "Handle must be a real uint64 scalar.");

            ObjectHandle<BaseClass> *handle = reinterpret_cast<ObjectHandle<BaseClass> *>(*((uint64_t*) mxGetData(in)));

            if (!handle->isValid())
                mexErrMsgIdAndTxt("MEX:ObjectHandle:invalidHandle", "Handle is not valid.");

            return handle;
        }

        uint32_t    mSignature;
        const std::string mName;
        BaseClass* const mPointer;
        const bool mOwner;
    };
}

#endif
