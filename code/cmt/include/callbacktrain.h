#ifndef CALLBACKTRAIN_H
#define CALLBACKTRAIN_H

#include <Python.h>
#include "conditionaldistribution.h"

struct CDObject;

class CallbackTrain : public ConditionalDistribution::Callback {
	public:
		CallbackTrain(CDObject* cd, PyObject* callback);
		CallbackTrain(const CallbackTrain& callbackTrain);
		virtual ~CallbackTrain();
		virtual CallbackTrain& operator=(const CallbackTrain& callbackTrain);
		virtual CallbackTrain* copy();
		virtual bool operator()(int iter, const ConditionalDistribution&);

	private:
		CDObject* mCD;
		PyObject* mCallback;
};

#endif
