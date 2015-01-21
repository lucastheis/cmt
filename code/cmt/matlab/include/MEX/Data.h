#ifndef __MEX_DATA_H__
#define __MEX_DATA_H__

#include "Output.h"
#include "Input.h"

namespace MEX {
    class Data : public Output {
    public:
        Data(int size);

        Data(const Data& other);

        ~Data();

        void resize(int size, bool fromFront = false);

        Data & operator=(const Data&);

        operator mxArray** const ();

        // ToDo: Fix this. There must be a smart way to merge Setter and Getter.
        MEX::Input::Getter operator()(int index) const;
    };
}

#endif
