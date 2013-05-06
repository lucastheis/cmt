#include "exception.h"
#include "callbackinterface.h"
#include "conditionaldistributioninterface.h"

#include "trainable.h"
using CMT::Trainable;

CallbackInterface::CallbackInterface(PyTypeObject* type, PyObject* callback) : 
	mType(type),
	mCallback(callback) 
{
	Py_INCREF(mCallback);
}



CallbackInterface::CallbackInterface(const CallbackInterface& callbackInterface) :
	mType(callbackInterface.mType),
	mCallback(callbackInterface.mCallback)
{
	Py_INCREF(mCallback);
}



CallbackInterface::~CallbackInterface() {
	Py_DECREF(mCallback);
}



CallbackInterface& CallbackInterface::operator=(const CallbackInterface& callbackInterface) {
	mType = callbackInterface.mType;

	Py_DECREF(mCallback);
	mCallback = callbackInterface.mCallback;
	Py_INCREF(mCallback);

	return *this;
}



CallbackInterface* CallbackInterface::copy() {
	return new CallbackInterface(*this);
}



bool CallbackInterface::operator()(int iter, const Trainable& cd) {
	CDObject* cdObj = reinterpret_cast<CDObject*>(CD_new(mType, 0, 0));

	// TODO: fix this hack
	cdObj->cd = const_cast<Trainable*>(&cd);
	cdObj->owner = false;

	// call Python object
	PyObject* args = Py_BuildValue("(iO)", iter, cdObj);
	PyObject* result = PyObject_CallObject(mCallback, args);
	Py_DECREF(args);

	// if cont is false, training will be aborted
	bool cont = true;

	if(result) {
		if(PyBool_Check(result))
			cont = (result == Py_True);
		Py_DECREF(result);
		Py_DECREF(cdObj);
	} else {
		Py_DECREF(cdObj);
		throw Exception("Some error occured during call to callback function.");
	}

	return cont;
}
