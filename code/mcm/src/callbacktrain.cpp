#include "callbacktrain.h"

CallbackTrain::CallbackTrain(MCGSMObject* mcgsm, PyObject* callback) : 
	mMcgsm(mcgsm), 
	mCallback(callback) 
{
	Py_INCREF(mMcgsm);
	Py_INCREF(mCallback);
}



CallbackTrain::CallbackTrain(const CallbackTrain& callbackTrain) :
	mMcgsm(callbackTrain.mMcgsm),
	mCallback(callbackTrain.mCallback)
{
	Py_INCREF(mMcgsm);
	Py_INCREF(mCallback);
}



CallbackTrain::~CallbackTrain() {
	Py_DECREF(mMcgsm);
	Py_DECREF(mCallback);
}



CallbackTrain& CallbackTrain::operator=(const CallbackTrain& callbackTrain) {
	Py_DECREF(mMcgsm);
	Py_DECREF(mCallback);

	mMcgsm = callbackTrain.mMcgsm;
	mCallback = callbackTrain.mCallback;

	Py_INCREF(mMcgsm);
	Py_INCREF(mCallback);

	return *this;
}



CallbackTrain* CallbackTrain::copy() {
	return new CallbackTrain(*this);
}



bool CallbackTrain::operator()(int iter, const MCGSM&) {
	// call Python object
	PyObject* args = Py_BuildValue("(iO)", iter, mMcgsm);
	PyObject* result = PyObject_CallObject(mCallback, args);

	Py_DECREF(args);

	// if cont is false, training will be aborted
	bool cont = true;
	if(result) {
		if(PyBool_Check(result))
			cont = (result == Py_True);
		Py_DECREF(result);
	} else {
		throw Exception("Some error occured during call to callback function.");
	}

	return cont;
}
