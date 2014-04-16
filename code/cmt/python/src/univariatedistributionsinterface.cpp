#include "univariatedistributionsinterface.h"

#include "cmt/utils"
using CMT::Exception;

const char* UnivariateDistribution_doc =
	"Abstract base class for univariate distributions used, for example, by L{GLM}.";

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



const char* Poisson_doc =
	"The Poisson distribution.\n"
	"\n"
	"$$p(k) = \\frac{\\lambda^k}{k!} \\exp(-\\lambda)$$\n"
	"\n"
	"@type  lambda: C{float}\n"
	"@param lambda: parameter of the Poisson distribution, $\\lambda$";

int Poisson_init(PoissonObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"lambda", 0};

	double lambda = 1.;

	if(!PyArg_ParseTupleAndKeywords(args, kwds, "|d", const_cast<char**>(kwlist), &lambda))
		return -1;

	try {
		self->distribution = new Poisson(lambda);
		return 0;
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	return -1;
}



const char* Poisson_reduce_doc =
	"__reduce__(self)\n"
	"\n"
	"Method used by Pickle.";

PyObject* Poisson_reduce(PoissonObject* self, PyObject*) {
	PyObject* args = Py_BuildValue("(d)", self->distribution->mean());
	PyObject* result = Py_BuildValue("(OO)", Py_TYPE(self), args);

	Py_DECREF(args);

	return result;
}



const char* Binomial_doc =
	"The binomial distribution.\n"
	"\n"
	"$$p(k) = \\binom{n}{k} p^k (1 - p)^{n - k}$$\n"
	"\n"
	"@type  n: C{float}\n"
	"@param n: parameter of the binomial distribution, $n$"
	"\n"
	"@type  p: C{float}\n"
	"@param p: parameter of the binomial distribution, $p$";

int Binomial_init(BinomialObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"n", "p", 0};

	int n = 10;
	double p = .5;

	if(!PyArg_ParseTupleAndKeywords(args, kwds, "|id", const_cast<char**>(kwlist), &n, &p))
		return -1;

	try {
		self->distribution = new Binomial(n, p);
		return 0;
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return -1;
	}

	return -1;
}



const char* Binomial_reduce_doc =
	"__reduce__(self)\n"
	"\n"
	"Method used by Pickle.";

PyObject* Binomial_reduce(BinomialObject* self, PyObject*) {
	PyObject* args = Py_BuildValue("(id)",
		self->distribution->number(),
		self->distribution->probability());
	PyObject* result = Py_BuildValue("(OO)", Py_TYPE(self), args);

	Py_DECREF(args);

	return result;
}
