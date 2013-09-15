#ifndef CALLBACKTRAIN_H
#define CALLBACKTRAIN_H

#include <Python.h>

#include "cmt/models"
using CMT::Trainable;

class CallbackInterface : public Trainable::Callback {
	public:
		CallbackInterface(PyTypeObject* type, PyObject* callback);
		CallbackInterface(const CallbackInterface& callbackInterface);
		virtual ~CallbackInterface();

		virtual CallbackInterface* copy();

		virtual CallbackInterface& operator=(const CallbackInterface& callbackInterface);
		virtual bool operator()(int iter, const CMT::Trainable&);

	private:
		PyTypeObject* mType;
		PyObject* mCallback;
};

#endif
