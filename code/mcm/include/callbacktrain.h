#ifndef CALLBACKTRAIN_H
#define CALLBACKTRAIN_H

#include <Python.h>
#include "mcgsm.h"

struct MCGSMObject;

class CallbackTrain : public MCGSM::Callback {
	public:
		CallbackTrain(MCGSMObject* mcgsm, PyObject* callback);
		CallbackTrain(const CallbackTrain& callbackTrain);
		virtual ~CallbackTrain();
		virtual CallbackTrain& operator=(const CallbackTrain& callbackTrain);
		virtual CallbackTrain* copy();
		virtual bool operator()(int iter, const MCGSM&);

	private:
		MCGSMObject* mMCGSM;
		PyObject* mCallback;
};

#endif
