#include "exception.h"
#include "callbacktrain.h"

CallbackTrain::CallbackTrain(CDObject* cd, PyObject* callback) : 
	mCD(cd),
	mCallback(callback) 
{
	Py_INCREF(mCD);
	Py_INCREF(mCallback);
}



CallbackTrain::CallbackTrain(const CallbackTrain& callbackTrain) :
	mCD(callbackTrain.mCD),
	mCallback(callbackTrain.mCallback)
{
	Py_INCREF(mCD);
	Py_INCREF(mCallback);
}



CallbackTrain::~CallbackTrain() {
	Py_DECREF(mCD);
	Py_DECREF(mCallback);
}



CallbackTrain& CallbackTrain::operator=(const CallbackTrain& callbackTrain) {
	Py_DECREF(mCD);
	Py_DECREF(mCallback);

	mCD = callbackTrain.mCD;
	mCallback = callbackTrain.mCallback;

	Py_INCREF(mCD);
	Py_INCREF(mCallback);

	return *this;
}



CallbackTrain* CallbackTrain::copy() {
	return new CallbackTrain(*this);
}



bool CallbackTrain::operator()(int iter, const ConditionalDistribution& cd) {
	// call Python object
	PyObject* args = Py_BuildValue("(iO)", iter, mCD);
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
