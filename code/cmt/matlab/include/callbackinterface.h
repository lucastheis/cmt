#ifndef __TRAINABLE_CALLBACK_H__
#define __TRAINABLE_CALLBACK_H__

#include "MEX/Function.h"

#include "trainable.h"

class TrainableCallback : public CMT::Trainable::Callback {
    public:
        TrainableCallback(MEX::Function function) : mFunction(function) {
        }

        virtual TrainableCallback* copy() {
            return new TrainableCallback(mFunction);
        }

        virtual bool operator()(int iter, const CMT::Trainable&);

    private:
        MEX::Function mFunction;
};

#endif
