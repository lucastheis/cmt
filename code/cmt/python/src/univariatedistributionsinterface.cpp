#include "univariatedistributionsinterface.h"

#include "cmt/utils"
using CMT::Exception;

const char* UnivariateDistribution_doc =
	"Abstract base class for univariate distributions usable by L{GLM}.";

int UnivariateDistribution_init(
	UnivariateDistributionObject* self,
	PyObject* args,
	PyObject* kwds)
{
	PyErr_SetString(PyExc_NotImplementedError, "This is an abstract class.");
	return -1;
}



const char* Bernoulli_doc =
	"The Bernoulli distribution.\n"
	"\n"
	"$$p(y) = \\rho^y (1 - \\rho)^{1 - y}$$\n"
	"\n"
	"@type  probability: C{float}\n"
	"@param probability: probability of generating a one, $\\rho$";

int Bernoulli_init(BernoulliObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"probability", 0};

	double prob = 0.5;

	if(!PyArg_ParseTupleAndKeywords(args, kwds, "|d", const_cast<char**>(kwlist), &prob))
		return -1;

	try {
		self->distribution = new Bernoulli(prob);
		return 0;
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	return -1;
}



const char* Bernoulli_reduce_doc =
	"__reduce__(self)\n"
	"\n"
	"Method used by Pickle.";

PyObject* Bernoulli_reduce(BernoulliObject* self, PyObject*) {
	PyObject* args = Py_BuildValue("(d)", self->distribution->probability());
	PyObject* result = Py_BuildValue("(OO)", Py_TYPE(self), args);

	Py_DECREF(args);

	return result;
}
