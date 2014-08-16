#include "MEX/Data.h"

MEX::Data::Data(int size) {
    mSize = size;
    mData =  (mxArray**) mxCalloc(mSize, sizeof(mxArray*));
}

MEX::Data::Data(const Data& other) {
    mSize = other.mSize;
    mData = (mxArray**) mxCalloc(mSize, sizeof(mxArray*));

    for(int i = 0; i < mSize; i++) {
        if(other.mData[i] != NULL) {
            mData[i] = mxDuplicateArray(other.mData[i]);
        }
    }
}

MEX::Data::~Data() {
    for(int i = 0; i < mSize; i++) {
        if(mData[i] != NULL) {
            mxDestroyArray(mData[i]);
            mData[i] = NULL;
        }
    }
    mxFree(mData);
}


MEX::Data& MEX::Data::operator=(const MEX::Data& other) {
    // Free all old data
    for(int i = 0; i < mSize; i++) {
        if(mData[i] != NULL) {
            mxDestroyArray(mData[i]);
            mData[i] = NULL;
        }
    }

    // Resize array to new size and zero it.
    mSize = other.mSize;
    mData = (mxArray**) mxRealloc(mData, mSize * sizeof(mxArray*));
    memset(mData, 0, mSize * sizeof(mxArray*));

    // Copy new data
    for(int i = 0; i < mSize; i++) {
        if(other.mData[i] != NULL) {
            mData[i] = mxDuplicateArray(other.mData[i]);
        }
    }
    return (*this);
}


MEX::Input::Getter MEX::Data::operator() (int index) const {
    return MEX::Input::Getter(mData[index], index);
}

void MEX::Data::resize(int size, bool fromFront) {
    int size_diff = size - mSize;

    // Free data that will be lost during resizing
    if (size_diff < 0) {
        if(fromFront) {
            for(int i = 0; i < -size_diff; i++) {
                if(mData[i] != NULL) {
                    mxDestroyArray(mData[i]);
                    mData[i] = NULL;
                }
            }
        } else {
            for(int i = size; i < mSize; i++) {
                if(mData[i] != NULL) {
                    mxDestroyArray(mData[i]);
                    mData[i] = NULL;
                }
            }
        }
    }

    // Resize the underlying array
    if(size_diff < 0 && fromFront) {
        // Remove data in front of array.
        mxArray** oldData = mData;

        mData = (mxArray**) mxCalloc(size, sizeof(mxArray*));

        memcpy(mData, oldData - size_diff, size);

        mxFree(oldData);
    } else {
        // Default method for all other cases
        mData = (mxArray**) mxRealloc(mData, size * sizeof(mxArray*));
    }

    // Initialize new space and fix positions if inFront is true
    if(size_diff > 0) {
        if(fromFront) {
            // Move data to the end of allocated space if requested
            memmove(mData + size_diff, mData, mSize * sizeof(mxArray*));
            // Zero new fields
            memset(mData, 0, size_diff * sizeof(mxArray*));
        } else {
            // Zero new fields
            memset(mData + mSize, 0, size_diff * sizeof(mxArray*));
        }
    }

    // Update size
    mSize = size;
}

MEX::Data::operator mxArray** const () {
    return mData;
}
