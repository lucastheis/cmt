#ifndef __TRAINABLE_CALLBACK_H__
#define __TRAINABLE_CALLBACK_H__

#include "MEX/Function.h"

#include "trainable.h"

template <class TrainableClass>
class TrainableCallback : public CMT::Trainable::Callback {
    public:
        TrainableCallback(MEX::Function constructor, MEX::Function function) : mConstrutor(constructor), mFunction(function) {
        }

        virtual TrainableCallback* copy() {
            return new TrainableCallback(* this);
        }

		virtual bool operator()(int iter, const CMT::Trainable& obj) {

			const TrainableClass& trainable = dynamic_cast<const TrainableClass&>(obj);

		    // Create arg list for
		    MEX::Data handle(1);
		    handle[0] = MEX::ObjectHandle<const TrainableClass>::share(&trainable);

		    // Construct object
		    MEX::Data args = mConstrutor(1, handle);

		    // Add iter to arguments
		    args.resize(2, true);
		    args[0] = iter;

		    // Call callback function
		    const MEX::Data& result = mFunction(1, args);

		    return result(0);
		}

    private:
        MEX::Function mConstrutor;
        MEX::Function mFunction;

};

#endif
