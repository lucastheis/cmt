#ifndef CALLBACKTRAIN_H
#define CALLBACKTRAIN_H

#include <Python.h>
#include "conditionaldistribution.h"
#include "conditionaldistributioninterface.h"

class CallbackInterface : public ConditionalDistribution::Callback {
	public:
		CallbackInterface(PyTypeObject* type, PyObject* callback);
		CallbackInterface(const CallbackInterface& callbackTrain);
		virtual ~CallbackInterface();
		virtual CallbackInterface& operator=(const CallbackInterface& callbackTrain);
		virtual CallbackInterface* copy();
		virtual bool operator()(int iter, const ConditionalDistribution&);

	private:
		PyTypeObject* mType;
		PyObject* mCallback;
};

#endif
