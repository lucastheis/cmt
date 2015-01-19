 #ifndef __MEX_CLASS_HANDLE_H__
#define __MEX_CLASS_HANDLE_H__

#include <typeinfo>
#include <string>
#include <cstring>
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
            return toMxArray(ptr, true);
        }

        static BaseClass* unwrap(const mxArray *in) {
            ObjectHandle* handle = fromValidMxArray(in);
            return handle->getPointer();
        }

        static bool validate(const mxArray *in) {
            ObjectHandle* handle = fromMxArray(in);

            if(!handle)
                return false;

            return handle->isValid();
        }

        static void free(const mxArray *in) {
            delete fromValidMxArray(in);
        }

    private:
        ObjectHandle(BaseClass* pointer, bool owner) : mPointer(pointer),
                                                       mName(typeid(BaseClass).name()),
                                                       mSignature(MEX_OBJECT_HANDLE_SIGNATURE),
                                                       mOwner(owner) {
            if(mOwner) mexLock();
        }

        ~ObjectHandle() {
            mSignature = 0;
            if(mOwner) {
                delete mPointer;
                mexUnlock();
            }
        }

        bool isValid() {
            if(mSignature != MEX_OBJECT_HANDLE_SIGNATURE) {
                mexWarnMsgIdAndTxt("mexWrapper:ObjectHandle:signature", "Invalid signature.");
            }
            if(strcmp(mName.c_str(), typeid(BaseClass).name())){
                mexWarnMsgIdAndTxt("mexWrapper:ObjectHandle:classMissmatch", "Class mismatch '%s' should be '%s'.", typeid(BaseClass).name(), mName.c_str());
            }

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

        static ObjectHandle* fromValidMxArray(const mxArray * in) {
            ObjectHandle* handle = fromMxArray(in);

            if(!handle)
                mexErrMsgIdAndTxt("MEX:ObjectHandle:notAHandle", "Handle must be a real uint64 scalar.");

            if (!handle->isValid())
                mexErrMsgIdAndTxt("MEX:ObjectHandle:invalidHandle", "Handle is not valid.");

            return handle;
        }

        static ObjectHandle* fromMxArray(const mxArray *in) {
            if (mxGetNumberOfElements(in) != 1 || !mxIsUint64(in) || mxIsComplex(in))
                return NULL;

            return reinterpret_cast<ObjectHandle<BaseClass> *>(*((uint64_t*) mxGetData(in)));
        }

        uint32_t    mSignature;
        const std::string mName;
        BaseClass* const mPointer;
        const bool mOwner;
    };
}

#endif
