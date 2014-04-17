#define PY_ARRAY_UNIQUE_SYMBOL CMT_ARRAY_API

#include <Python.h>
#include <arrayobject.h>
#include <structmember.h>
#include <stdlib.h>
#include <sys/time.h>
#include "conditionaldistributioninterface.h"
#include "distribution.h"
#include "distributioninterface.h"
#include "fvbninterface.h"
#include "glminterface.h"
#include "gsminterface.h"
#include "mcbminterface.h"
#include "mcgsminterface.h"
#include "mixtureinterface.h"
#include "mlrinterface.h"
#include "nonlinearitiesinterface.h"
#include "patchmodelinterface.h"
#include "preconditionerinterface.h"
#include "stminterface.h"
#include "univariatedistributionsinterface.h"
#include "toolsinterface.h"
#include "trainableinterface.h"
#include "Eigen/Core"

static PyGetSetDef Distribution_getset[] = {
	{"dim", (getter)Distribution_dim, 0, "Dimensionality of the distribution."},
	{0}
};

static PyMethodDef Distribution_methods[] = {
	{"sample", (PyCFunction)Distribution_sample, METH_VARARGS | METH_KEYWORDS, Distribution_sample_doc},
	{"loglikelihood",
		(PyCFunction)Distribution_loglikelihood,
		METH_VARARGS | METH_KEYWORDS,
		Distribution_loglikelihood_doc},
	{"evaluate",
		(PyCFunction)Distribution_evaluate,
		METH_VARARGS | METH_KEYWORDS,
		Distribution_evaluate_doc},
	{0}
};

PyTypeObject Distribution_type = {
	PyObject_HEAD_INIT(0)
	0,                                    /*ob_size*/
	"cmt.models.ConditionalDistribution", /*tp_name*/
	sizeof(DistributionObject),           /*tp_basicsize*/
	0,                                    /*tp_itemsize*/
	(destructor)Distribution_dealloc,     /*tp_dealloc*/
	0,                                    /*tp_print*/
	0,                                    /*tp_getattr*/
	0,                                    /*tp_setattr*/
	0,                                    /*tp_compare*/
	0,                                    /*tp_repr*/
	0,                                    /*tp_as_number*/
	0,                                    /*tp_as_sequence*/
	0,                                    /*tp_as_mapping*/
	0,                                    /*tp_hash */
	0,                                    /*tp_call*/
	0,                                    /*tp_str*/
	0,                                    /*tp_getattro*/
	0,                                    /*tp_setattro*/
	0,                                    /*tp_as_buffer*/
	Py_TPFLAGS_DEFAULT,                   /*tp_flags*/
	Distribution_doc,                     /*tp_doc*/
	0,                                    /*tp_traverse*/
	0,                                    /*tp_clear*/
	0,                                    /*tp_richcompare*/
	0,                                    /*tp_weaklistoffset*/
	0,                                    /*tp_iter*/
	0,                                    /*tp_iternext*/
	Distribution_methods,                 /*tp_methods*/
	0,                                    /*tp_members*/
	Distribution_getset,                  /*tp_getset*/
	0,                                    /*tp_base*/
	0,                                    /*tp_dict*/
	0,                                    /*tp_descr_get*/
	0,                                    /*tp_descr_set*/
	0,                                    /*tp_dictoffset*/
	(initproc)Distribution_init,          /*tp_init*/
	0,                                    /*tp_alloc*/
	Distribution_new,                     /*tp_new*/
};

static PyGetSetDef CD_getset[] = {
	{"dim_in", (getter)CD_dim_in, 0, "Dimensionality of inputs."},
	{"dim_out", (getter)CD_dim_out, 0, "Dimensionality of outputs."},
	{0}
};

static PyMethodDef CD_methods[] = {
	{"sample", (PyCFunction)CD_sample, METH_VARARGS | METH_KEYWORDS, CD_sample_doc},
	{"predict", (PyCFunction)CD_predict, METH_VARARGS | METH_KEYWORDS, CD_predict_doc},
	{"loglikelihood",
		(PyCFunction)CD_loglikelihood,
		METH_VARARGS | METH_KEYWORDS,
		CD_loglikelihood_doc},
	{"evaluate",
		(PyCFunction)CD_evaluate,
		METH_VARARGS | METH_KEYWORDS,
		CD_evaluate_doc},
	{"_data_gradient",
		(PyCFunction)CD_data_gradient,
		METH_VARARGS | METH_KEYWORDS, 0},
	{0}
};

PyTypeObject CD_type = {
	PyObject_HEAD_INIT(0)
	0,                                    /*ob_size*/
	"cmt.models.ConditionalDistribution", /*tp_name*/
	sizeof(CDObject),                     /*tp_basicsize*/
	0,                                    /*tp_itemsize*/
	(destructor)CD_dealloc,               /*tp_dealloc*/
	0,                                    /*tp_print*/
	0,                                    /*tp_getattr*/
	0,                                    /*tp_setattr*/
	0,                                    /*tp_compare*/
	0,                                    /*tp_repr*/
	0,                                    /*tp_as_number*/
	0,                                    /*tp_as_sequence*/
	0,                                    /*tp_as_mapping*/
	0,                                    /*tp_hash */
	0,                                    /*tp_call*/
	0,                                    /*tp_str*/
	0,                                    /*tp_getattro*/
	0,                                    /*tp_setattro*/
	0,                                    /*tp_as_buffer*/
	Py_TPFLAGS_DEFAULT,                   /*tp_flags*/
	CD_doc,                               /*tp_doc*/
	0,                                    /*tp_traverse*/
	0,                                    /*tp_clear*/
	0,                                    /*tp_richcompare*/
	0,                                    /*tp_weaklistoffset*/
	0,                                    /*tp_iter*/
	0,                                    /*tp_iternext*/
	CD_methods,                           /*tp_methods*/
	0,                                    /*tp_members*/
	CD_getset,                            /*tp_getset*/
	0,                                    /*tp_base*/
	0,                                    /*tp_dict*/
	0,                                    /*tp_descr_get*/
	0,                                    /*tp_descr_set*/
	0,                                    /*tp_dictoffset*/
	(initproc)CD_init,                    /*tp_init*/
	0,                                    /*tp_alloc*/
	CD_new,                               /*tp_new*/
};

static PyGetSetDef MCGSM_getset[] = {
	{"num_components", (getter)MCGSM_num_components, 0, "Numer of predictors."},
	{"num_scales", (getter)MCGSM_num_scales, 0, "Number of scale variables per component."},
	{"num_features",
		(getter)MCGSM_num_features, 0,
		"Number of features available to approximate input covariances."},
	{"priors",
		(getter)MCGSM_priors,
		(setter)MCGSM_set_priors,
		"Log-weights of mixture components and scales, $\\eta_{cs}$."},
	{"scales",
		(getter)MCGSM_scales,
		(setter)MCGSM_set_scales,
		"Log-precision variables, $\\alpha_{cs}$."},
	{"weights",
		(getter)MCGSM_weights,
		(setter)MCGSM_set_weights,
		"Weights relating features and mixture components, $\\beta_{ci}$."},
	{"features", 
		(getter)MCGSM_features,
		(setter)MCGSM_set_features,
		"Features used for capturing structure in inputs, $\\mathbf{b}_i$."},
	{"cholesky_factors", 
		(getter)MCGSM_cholesky_factors, 
		(setter)MCGSM_set_cholesky_factors, 
		"A list of Cholesky factors of residual precision matrices, $\\mathbf{L}_c$."},
	{"predictors",
		(getter)MCGSM_predictors,
		(setter)MCGSM_set_predictors,
		"A list of linear predictors, $\\mathbf{A}_c$."},
	{"linear_features",
		(getter)MCGSM_linear_features,
		(setter)MCGSM_set_linear_features,
		"Linear features, $\\mathbf{w}_c$."},
	{"means",
		(getter)MCGSM_means,
		(setter)MCGSM_set_means,
		"Means of outputs, $\\mathbf{u}_c$."},
	{0}
};

static PyMethodDef MCGSM_methods[] = {
	{"initialize",
		(PyCFunction)Trainable_initialize,
		METH_VARARGS | METH_KEYWORDS,
		Trainable_initialize_doc},
	{"train", (PyCFunction)MCGSM_train, METH_VARARGS | METH_KEYWORDS, MCGSM_train_doc},
	{"prior",
		(PyCFunction)MCGSM_prior,
		METH_VARARGS | METH_KEYWORDS,
		MCGSM_prior_doc},
	{"posterior",
		(PyCFunction)MCGSM_posterior,
		METH_VARARGS | METH_KEYWORDS,
		MCGSM_posterior_doc},
	{"loglikelihood",
		(PyCFunction)MCGSM_loglikelihood,
		METH_VARARGS | METH_KEYWORDS,
		MCGSM_loglikelihood_doc},
	{"sample",
		(PyCFunction)MCGSM_sample,
		METH_VARARGS | METH_KEYWORDS,
		MCGSM_sample_doc},
	{"sample_prior",
		(PyCFunction)MCGSM_sample_prior,
		METH_VARARGS | METH_KEYWORDS,
		MCGSM_sample_prior_doc},
	{"sample_posterior",
		(PyCFunction)MCGSM_sample_posterior,
		METH_VARARGS | METH_KEYWORDS,
		MCGSM_sample_posterior_doc},
	{"_check_gradient",
		(PyCFunction)MCGSM_check_gradient,
		METH_VARARGS | METH_KEYWORDS,
		Trainable_check_gradient_doc},
	{"_check_performance",
		(PyCFunction)MCGSM_check_performance,
		METH_VARARGS | METH_KEYWORDS, 
		Trainable_check_performance_doc},
	{"_parameters",
		(PyCFunction)MCGSM_parameters,
		METH_VARARGS | METH_KEYWORDS,
		Trainable_parameters_doc},
	{"_set_parameters",
		(PyCFunction)MCGSM_set_parameters,
		METH_VARARGS | METH_KEYWORDS,
		Trainable_set_parameters_doc},
	{"_parameter_gradient",
		(PyCFunction)MCGSM_parameter_gradient,
		METH_VARARGS | METH_KEYWORDS,
		Trainable_parameter_gradient_doc},
	{"__reduce__", (PyCFunction)MCGSM_reduce, METH_NOARGS, MCGSM_reduce_doc},
	{"__setstate__", (PyCFunction)MCGSM_setstate, METH_VARARGS, MCGSM_setstate_doc},
	{0}
};

PyTypeObject MCGSM_type = {
	PyObject_HEAD_INIT(0)
	0,                      /*ob_size*/
	"cmt.models.MCGSM",     /*tp_name*/
	sizeof(MCGSMObject),    /*tp_basicsize*/
	0,                      /*tp_itemsize*/
	(destructor)CD_dealloc, /*tp_dealloc*/
	0,                      /*tp_print*/
	0,                      /*tp_getattr*/
	0,                      /*tp_setattr*/
	0,                      /*tp_compare*/
	0,                      /*tp_repr*/
	0,                      /*tp_as_number*/
	0,                      /*tp_as_sequence*/
	0,                      /*tp_as_mapping*/
	0,                      /*tp_hash */
	0,                      /*tp_call*/
	0,                      /*tp_str*/
	0,                      /*tp_getattro*/
	0,                      /*tp_setattro*/
	0,                      /*tp_as_buffer*/
	Py_TPFLAGS_DEFAULT,     /*tp_flags*/
	MCGSM_doc,              /*tp_doc*/
	0,                      /*tp_traverse*/
	0,                      /*tp_clear*/
	0,                      /*tp_richcompare*/
	0,                      /*tp_weaklistoffset*/
	0,                      /*tp_iter*/
	0,                      /*tp_iternext*/
	MCGSM_methods,          /*tp_methods*/
	0,                      /*tp_members*/
	MCGSM_getset,           /*tp_getset*/
	&CD_type,               /*tp_base*/
	0,                      /*tp_dict*/
	0,                      /*tp_descr_get*/
	0,                      /*tp_descr_set*/
	0,                      /*tp_dictoffset*/
	(initproc)MCGSM_init,   /*tp_init*/
	0,                      /*tp_alloc*/
	CD_new,                 /*tp_new*/
};

static PyGetSetDef MCBM_getset[] = {
	{"num_components", (getter)MCBM_num_components, 0, "Numer of predictors."},
	{"num_features",
		(getter)MCBM_num_features, 0,
		"Number of features available to approximate input covariances."},
	{"priors",
		(getter)MCBM_priors,
		(setter)MCBM_set_priors,
		"Log-weights of mixture components and scales, $\\eta_{cs}$."},
	{"weights",
		(getter)MCBM_weights,
		(setter)MCBM_set_weights,
		"Weights relating features and mixture components, $\\beta_{ci}$."},
	{"features", 
		(getter)MCBM_features,
		(setter)MCBM_set_features,
		"Features used for capturing structure in inputs, $\\mathbf{b}_i$."},
	{"predictors",
		(getter)MCBM_predictors,
		(setter)MCBM_set_predictors,
		"Parameters relating inputs and outputs, $\\mathbf{A}_c$."},
	{"input_bias",
		(getter)MCBM_input_bias,
		(setter)MCBM_set_input_bias,
		"Input bias vectors, $\\mathbf{w}_c$."},
	{"output_bias",
		(getter)MCBM_output_bias,
		(setter)MCBM_set_output_bias,
		"Output biases, $v_c$."},
	{0}
};

static PyMethodDef MCBM_methods[] = {
	{"train", (PyCFunction)MCBM_train, METH_VARARGS | METH_KEYWORDS, MCBM_train_doc},
	{"sample_posterior",
		(PyCFunction)MCBM_sample_posterior,
		METH_VARARGS | METH_KEYWORDS,
		MCBM_sample_posterior_doc},
	{"_parameters",
		(PyCFunction)MCBM_parameters,
		METH_VARARGS | METH_KEYWORDS,
		Trainable_parameters_doc},
	{"_set_parameters",
		(PyCFunction)MCBM_set_parameters,
		METH_VARARGS | METH_KEYWORDS,
		Trainable_set_parameters_doc},
	{"_parameter_gradient",
		(PyCFunction)MCBM_parameter_gradient,
		METH_VARARGS | METH_KEYWORDS,
		Trainable_parameter_gradient_doc},
	{"_check_performance",
		(PyCFunction)MCBM_check_performance,
		METH_VARARGS | METH_KEYWORDS,
		Trainable_check_performance_doc},
	{"_check_gradient",
		(PyCFunction)MCBM_check_gradient,
		METH_VARARGS | METH_KEYWORDS,
		Trainable_check_gradient_doc},
	{"__reduce__", (PyCFunction)MCBM_reduce, METH_NOARGS, MCBM_reduce_doc},
	{"__setstate__", (PyCFunction)MCBM_setstate, METH_VARARGS, MCBM_setstate_doc},
	{0}
};

PyTypeObject MCBM_type = {
	PyObject_HEAD_INIT(0)
	0,                      /*ob_size*/
	"cmt.models.MCBM",      /*tp_name*/
	sizeof(MCBMObject),     /*tp_basicsize*/
	0,                      /*tp_itemsize*/
	(destructor)CD_dealloc, /*tp_dealloc*/
	0,                      /*tp_print*/
	0,                      /*tp_getattr*/
	0,                      /*tp_setattr*/
	0,                      /*tp_compare*/
	0,                      /*tp_repr*/
	0,                      /*tp_as_number*/
	0,                      /*tp_as_sequence*/
	0,                      /*tp_as_mapping*/
	0,                      /*tp_hash */
	0,                      /*tp_call*/
	0,                      /*tp_str*/
	0,                      /*tp_getattro*/
	0,                      /*tp_setattro*/
	0,                      /*tp_as_buffer*/
	Py_TPFLAGS_DEFAULT,     /*tp_flags*/
	MCBM_doc,               /*tp_doc*/
	0,                      /*tp_traverse*/
	0,                      /*tp_clear*/
	0,                      /*tp_richcompare*/
	0,                      /*tp_weaklistoffset*/
	0,                      /*tp_iter*/
	0,                      /*tp_iternext*/
	MCBM_methods,           /*tp_methods*/
	0,                      /*tp_members*/
	MCBM_getset,            /*tp_getset*/
	&CD_type,               /*tp_base*/
	0,                      /*tp_dict*/
	0,                      /*tp_descr_get*/
	0,                      /*tp_descr_set*/
	0,                      /*tp_dictoffset*/
	(initproc)MCBM_init,    /*tp_init*/
	0,                      /*tp_alloc*/
	CD_new,                 /*tp_new*/
};

static PyGetSetDef STM_getset[] = {
	{"dim_in_nonlinear", (getter)STM_dim_in_nonlinear, 0, "Dimensionality of nonlinear inputs."},
	{"dim_in_linear", (getter)STM_dim_in_linear, 0, "Dimensionality of linear inputs."},
	{"num_components", (getter)STM_num_components, 0, "Numer of predictors."},
	{"num_features",
		(getter)STM_num_features, 0,
		"Number of features available to approximate input covariances."},
	{"sharpness",
		(getter)STM_sharpness,
		(setter)STM_set_sharpness,
		"Controls the sharpness of the soft-maximum implemented by the log-sum-exp, $\\lambda$."},
	{"biases",
		(getter)STM_biases,
		(setter)STM_set_biases,
		"Bias terms controlling strength of each mixture component, $a_k$."},
	{"weights",
		(getter)STM_weights,
		(setter)STM_set_weights,
		"Weights relating features and mixture components, $\\beta_{kl}$."},
	{"features", 
		(getter)STM_features,
		(setter)STM_set_features,
		"Features used for capturing structure in inputs, $\\mathbf{u}_l$."},
	{"predictors",
		(getter)STM_predictors,
		(setter)STM_set_predictors,
		"Parameters relating inputs and outputs, $\\mathbf{w}_k$."},
	{"linear_predictor",
		(getter)STM_linear_predictor,
		(setter)STM_set_linear_predictor,
		"Parameters relating inputs and outputs, $\\mathbf{v}$."},
	{"nonlinearity",
		(getter)STM_nonlinearity,
		(setter)STM_set_nonlinearity,
		"Nonlinearity applied to output of log-sum-exp, $g$."},
	{"distribution",
		(getter)STM_distribution,
		(setter)STM_set_distribution,
		"Distribution whose average value is determined by output of nonlinearity."},
	{0}
};

static PyMethodDef STM_methods[] = {
	{"initialize",
		(PyCFunction)Trainable_initialize,
		METH_VARARGS | METH_KEYWORDS,
		Trainable_initialize_doc},
	{"linear_response", 
		(PyCFunction)STM_linear_response,
		METH_VARARGS | METH_KEYWORDS,
		STM_linear_response_doc},
	{"nonlinear_responses", 
		(PyCFunction)STM_nonlinear_responses,
		METH_VARARGS | METH_KEYWORDS,
		STM_nonlinear_responses_doc},
	{"train", (PyCFunction)STM_train, METH_VARARGS | METH_KEYWORDS, STM_train_doc},
	{"_parameters",
		(PyCFunction)STM_parameters,
		METH_VARARGS | METH_KEYWORDS,
		Trainable_parameters_doc},
	{"_set_parameters",
		(PyCFunction)STM_set_parameters,
		METH_VARARGS | METH_KEYWORDS,
		Trainable_set_parameters_doc},
	{"_parameter_gradient",
		(PyCFunction)STM_parameter_gradient,
		METH_VARARGS | METH_KEYWORDS,
		Trainable_parameter_gradient_doc},
	{"_fisher_information",
		(PyCFunction)STM_fisher_information,
		METH_VARARGS | METH_KEYWORDS, 
		Trainable_fisher_information_doc},
	{"_check_performance",
		(PyCFunction)STM_check_performance,
		METH_VARARGS | METH_KEYWORDS,
		Trainable_check_performance_doc},
	{"_check_gradient",
		(PyCFunction)STM_check_gradient,
		METH_VARARGS | METH_KEYWORDS,
		Trainable_check_gradient_doc},
	{"__reduce__", (PyCFunction)STM_reduce, METH_NOARGS, STM_reduce_doc},
	{"__setstate__", (PyCFunction)STM_setstate, METH_VARARGS, STM_setstate_doc},
	{0}
};

PyTypeObject STM_type = {
	PyObject_HEAD_INIT(0)
	0,                      /*ob_size*/
	"cmt.models.STM",       /*tp_name*/
	sizeof(STMObject),      /*tp_basicsize*/
	0,                      /*tp_itemsize*/
	(destructor)CD_dealloc, /*tp_dealloc*/
	0,                      /*tp_print*/
	0,                      /*tp_getattr*/
	0,                      /*tp_setattr*/
	0,                      /*tp_compare*/
	0,                      /*tp_repr*/
	0,                      /*tp_as_number*/
	0,                      /*tp_as_sequence*/
	0,                      /*tp_as_mapping*/
	0,                      /*tp_hash */
	0,                      /*tp_call*/
	0,                      /*tp_str*/
	0,                      /*tp_getattro*/
	0,                      /*tp_setattro*/
	0,                      /*tp_as_buffer*/
	Py_TPFLAGS_DEFAULT,     /*tp_flags*/
	STM_doc,                /*tp_doc*/
	0,                      /*tp_traverse*/
	0,                      /*tp_clear*/
	0,                      /*tp_richcompare*/
	0,                      /*tp_weaklistoffset*/
	0,                      /*tp_iter*/
	0,                      /*tp_iternext*/
	STM_methods,            /*tp_methods*/
	0,                      /*tp_members*/
	STM_getset,             /*tp_getset*/
	&CD_type,               /*tp_base*/
	0,                      /*tp_dict*/
	0,                      /*tp_descr_get*/
	0,                      /*tp_descr_set*/
	0,                      /*tp_dictoffset*/
	(initproc)STM_init,     /*tp_init*/
	0,                      /*tp_alloc*/
	CD_new,                 /*tp_new*/
};

static PyMappingMethods Mixture_as_mapping = {
	0,                             /*mp_length*/
	(binaryfunc)Mixture_subscript, /*mp_subscript*/
	0,                             /*mp_ass_subscript*/
};

static PyGetSetDef Mixture_getset[] = {
	{"priors",
		(getter)Mixture_priors,
		(setter)Mixture_set_priors,
		"Prior probabilities of mixture components, $\\pi_k$."},
	{"num_components",
		(getter)Mixture_num_components,
		0,
		"Number of mixture components."},
	{0}
};

static PyMethodDef Mixture_methods[] = {
	{"train", (PyCFunction)Mixture_train, METH_VARARGS | METH_KEYWORDS, Mixture_train_doc},
	{"initialize", (PyCFunction)Mixture_initialize, METH_VARARGS | METH_KEYWORDS, Mixture_initialize_doc},
	{"add_component", (PyCFunction)Mixture_add_component, METH_VARARGS | METH_KEYWORDS, 0},
	{"__reduce__", (PyCFunction)Mixture_reduce, METH_NOARGS, 0},
	{"__setstate__", (PyCFunction)Mixture_setstate, METH_VARARGS, 0},
	{0}
};

PyTypeObject Mixture_type = {
	PyObject_HEAD_INIT(0)
	0,                                /*ob_size*/
	"cmt.models.Mixture",             /*tp_name*/
	sizeof(MixtureObject),            /*tp_basicsize*/
	0,                                /*tp_itemsize*/
	(destructor)Distribution_dealloc, /*tp_dealloc*/
	0,                                /*tp_print*/
	0,                                /*tp_getattr*/
	0,                                /*tp_setattr*/
	0,                                /*tp_compare*/
	0,                                /*tp_repr*/
	0,                                /*tp_as_number*/
	0,                                /*tp_as_sequence*/
	&Mixture_as_mapping,              /*tp_as_mapping*/
	0,                                /*tp_hash */
	0,                                /*tp_call*/
	0,                                /*tp_str*/
	0,                                /*tp_getattro*/
	0,                                /*tp_setattro*/
	0,                                /*tp_as_buffer*/
	Py_TPFLAGS_DEFAULT,               /*tp_flags*/
	Mixture_doc,                      /*tp_doc*/
	0,                                /*tp_traverse*/
	0,                                /*tp_clear*/
	0,                                /*tp_richcompare*/
	0,                                /*tp_weaklistoffset*/
	0,                                /*tp_iter*/
	0,                                /*tp_iternext*/
	Mixture_methods,                  /*tp_methods*/
	0,                                /*tp_members*/
	Mixture_getset,                   /*tp_getset*/
	&Distribution_type,               /*tp_base*/
	0,                                /*tp_dict*/
	0,                                /*tp_descr_get*/
	0,                                /*tp_descr_set*/
	0,                                /*tp_dictoffset*/
	(initproc)Mixture_init,           /*tp_init*/
	0,                                /*tp_alloc*/
	Distribution_new,                 /*tp_new*/
};

static PyGetSetDef MoGSM_getset[] = {
	{"num_scales",
		(getter)MoGSM_num_scales,
		0,
		"Number of scales specified at construction."},
	{0}
};

static PyMethodDef MoGSM_methods[] = {
	{"__reduce__", (PyCFunction)MoGSM_reduce, METH_NOARGS, 0},
	{0},
};

PyTypeObject MoGSM_type = {
	PyObject_HEAD_INIT(0)
	0,                                /*ob_size*/
	"cmt.models.MoGSM",               /*tp_name*/
	sizeof(MoGSMObject),              /*tp_basicsize*/
	0,                                /*tp_itemsize*/
	(destructor)Distribution_dealloc, /*tp_dealloc*/
	0,                                /*tp_print*/
	0,                                /*tp_getattr*/
	0,                                /*tp_setattr*/
	0,                                /*tp_compare*/
	0,                                /*tp_repr*/
	0,                                /*tp_as_number*/
	0,                                /*tp_as_sequence*/
	0,                                /*tp_as_mapping*/
	0,                                /*tp_hash */
	0,                                /*tp_call*/
	0,                                /*tp_str*/
	0,                                /*tp_getattro*/
	0,                                /*tp_setattro*/
	0,                                /*tp_as_buffer*/
	Py_TPFLAGS_DEFAULT,               /*tp_flags*/
	MoGSM_doc,                        /*tp_doc*/
	0,                                /*tp_traverse*/
	0,                                /*tp_clear*/
	0,                                /*tp_richcompare*/
	0,                                /*tp_weaklistoffset*/
	0,                                /*tp_iter*/
	0,                                /*tp_iternext*/
	MoGSM_methods,                    /*tp_methods*/
	0,                                /*tp_members*/
	MoGSM_getset,                     /*tp_getset*/
	&Mixture_type,                    /*tp_base*/
	0,                                /*tp_dict*/
	0,                                /*tp_descr_get*/
	0,                                /*tp_descr_set*/
	0,                                /*tp_dictoffset*/
	(initproc)MoGSM_init,             /*tp_init*/
	0,                                /*tp_alloc*/
	Distribution_new,                 /*tp_new*/
};

static PyMethodDef MixtureComponent_methods[] = {
	{"train", (PyCFunction)MixtureComponent_train, METH_VARARGS | METH_KEYWORDS, MixtureComponent_train_doc},
	{"initialize", (PyCFunction)MixtureComponent_initialize, METH_VARARGS | METH_KEYWORDS, MixtureComponent_initialize_doc},
	{0}
};

PyTypeObject MixtureComponent_type = {
	PyObject_HEAD_INIT(0)
	0,                                /*ob_size*/
	"cmt.models.MixtureComponent",    /*tp_name*/
	sizeof(MixtureComponentObject),   /*tp_basicsize*/
	0,                                /*tp_itemsize*/
	(destructor)Distribution_dealloc, /*tp_dealloc*/
	0,                                /*tp_print*/
	0,                                /*tp_getattr*/
	0,                                /*tp_setattr*/
	0,                                /*tp_compare*/
	0,                                /*tp_repr*/
	0,                                /*tp_as_number*/
	0,                                /*tp_as_sequence*/
	0,                                /*tp_as_mapping*/
	0,                                /*tp_hash */
	0,                                /*tp_call*/
	0,                                /*tp_str*/
	0,                                /*tp_getattro*/
	0,                                /*tp_setattro*/
	0,                                /*tp_as_buffer*/
	Py_TPFLAGS_DEFAULT,               /*tp_flags*/
	MixtureComponent_doc,             /*tp_doc*/
	0,                                /*tp_traverse*/
	0,                                /*tp_clear*/
	0,                                /*tp_richcompare*/
	0,                                /*tp_weaklistoffset*/
	0,                                /*tp_iter*/
	0,                                /*tp_iternext*/
	MixtureComponent_methods,         /*tp_methods*/
	0,                                /*tp_members*/
	0,                                /*tp_getset*/
	&Distribution_type,               /*tp_base*/
	0,                                /*tp_dict*/
	0,                                /*tp_descr_get*/
	0,                                /*tp_descr_set*/
	0,                                /*tp_dictoffset*/
	(initproc)MixtureComponent_init,  /*tp_init*/
	0,                                /*tp_alloc*/
	Distribution_new,                 /*tp_new*/
};

static PyGetSetDef GSM_getset[] = {
	{"num_scales",
		(getter)GSM_num_scales,
		0,
		"Number of precision scale variables."},
	{"mean",
		(getter)GSM_mean,
		(setter)GSM_set_mean,
		"Mean of the distribution, $\\boldsymbol{\\mu}$."},
	{"priors",
		(getter)GSM_priors,
		(setter)GSM_set_priors,
		"Prior probabilities of scales, $\\pi_k$."},
	{"scales",
		(getter)GSM_scales,
		(setter)GSM_set_scales,
		"Precision scale variables of the GSM, $\\lambda_k$."},
	{"covariance",
		(getter)GSM_covariance,
		(setter)GSM_set_covariance,
		"Covariance matrix of the GSM, $\\boldsymbol{\\Sigma}$."},
	{0}
};

static PyMethodDef GSM_methods[] = {
	{"__reduce__", (PyCFunction)GSM_reduce, METH_NOARGS, 0},
	{"__setstate__", (PyCFunction)GSM_setstate, METH_VARARGS, 0},
	{0}
};

PyTypeObject GSM_type = {
	PyObject_HEAD_INIT(0)
	0,                                /*ob_size*/
	"cmt.models.GSM",                 /*tp_name*/
	sizeof(GSMObject),                /*tp_basicsize*/
	0,                                /*tp_itemsize*/
	(destructor)Distribution_dealloc, /*tp_dealloc*/
	0,                                /*tp_print*/
	0,                                /*tp_getattr*/
	0,                                /*tp_setattr*/
	0,                                /*tp_compare*/
	0,                                /*tp_repr*/
	0,                                /*tp_as_number*/
	0,                                /*tp_as_sequence*/
	0,                                /*tp_as_mapping*/
	0,                                /*tp_hash */
	0,                                /*tp_call*/
	0,                                /*tp_str*/
	0,                                /*tp_getattro*/
	0,                                /*tp_setattro*/
	0,                                /*tp_as_buffer*/
	Py_TPFLAGS_DEFAULT,               /*tp_flags*/
	GSM_doc,                          /*tp_doc*/
	0,                                /*tp_traverse*/
	0,                                /*tp_clear*/
	0,                                /*tp_richcompare*/
	0,                                /*tp_weaklistoffset*/
	0,                                /*tp_iter*/
	0,                                /*tp_iternext*/
	GSM_methods,                      /*tp_methods*/
	0,                                /*tp_members*/
	GSM_getset,                       /*tp_getset*/
	&MixtureComponent_type,           /*tp_base*/
	0,                                /*tp_dict*/
	0,                                /*tp_descr_get*/
	0,                                /*tp_descr_set*/
	0,                                /*tp_dictoffset*/
	(initproc)GSM_init,               /*tp_init*/
	0,                                /*tp_alloc*/
	Distribution_new,                 /*tp_new*/
};

static PyGetSetDef PatchModel_getset[] = {
	{"rows", (getter)PatchModel_rows, 0, "Number of rows of the modeled patches."},
	{"cols", (getter)PatchModel_cols, 0, "Number of columns of the modeled patches."},
	{"order", (getter)PatchModel_order, 0, "Order in which pixels are sampled."},
	{0}
};

static PyMethodDef PatchModel_methods[] = {
	{"loglikelihood", (PyCFunction)PatchModel_loglikelihood, METH_KEYWORDS, 0},
	{"input_mask", (PyCFunction)PatchModel_input_mask, METH_VARARGS, 0},
	{"output_mask", (PyCFunction)PatchModel_output_mask, METH_VARARGS, 0},
	{"input_indices", (PyCFunction)PatchModel_input_indices, 0, 0},
	{0}
};

PyTypeObject PatchModel_type = {
	PyObject_HEAD_INIT(0)
	0,                                /*ob_size*/
	"cmt.models.PatchModel",          /*tp_name*/
	sizeof(PatchModelObject),         /*tp_basicsize*/
	0,                                /*tp_itemsize*/
	(destructor)Distribution_dealloc, /*tp_dealloc*/
	0,                                /*tp_print*/
	0,                                /*tp_getattr*/
	0,                                /*tp_setattr*/
	0,                                /*tp_compare*/
	0,                                /*tp_repr*/
	0,                                /*tp_as_number*/
	0,                                /*tp_as_sequence*/
	0,                                /*tp_as_mapping*/
	0,                                /*tp_hash */
	0,                                /*tp_call*/
	0,                                /*tp_str*/
	0,                                /*tp_getattro*/
	0,                                /*tp_setattro*/
	0,                                /*tp_as_buffer*/
	Py_TPFLAGS_DEFAULT,               /*tp_flags*/
	PatchModel_doc,                   /*tp_doc*/
	0,                                /*tp_traverse*/
	0,                                /*tp_clear*/
	0,                                /*tp_richcompare*/
	0,                                /*tp_weaklistoffset*/
	0,                                /*tp_iter*/
	0,                                /*tp_iternext*/
	PatchModel_methods,               /*tp_methods*/
	0,                                /*tp_members*/
	PatchModel_getset,                /*tp_getset*/
	&Distribution_type,               /*tp_base*/
	0,                                /*tp_dict*/
	0,                                /*tp_descr_get*/
	0,                                /*tp_descr_set*/
	0,                                /*tp_dictoffset*/
	(initproc)Distribution_init,      /*tp_init*/
	0,                                /*tp_alloc*/
	Distribution_new,                 /*tp_new*/
};

static PyMappingMethods PatchMCBM_as_mapping = {
	0,                                      /*mp_length*/
	(binaryfunc)PatchMCBM_subscript,        /*mp_subscript*/
	(objobjargproc)PatchMCBM_ass_subscript, /*mp_ass_subscript*/
};

static PyGetSetDef PatchMCBM_getset[] = {
	{"preconditioners", 
		(getter)PatchMCBM_preconditioners,
		(setter)PatchMCBM_set_preconditioners,
		"A dictionary containing all preconditioners."},
	{0}
};

static PyMethodDef PatchMCBM_methods[] = {
	{"initialize", (PyCFunction)PatchMCBM_initialize, METH_KEYWORDS, PatchMCBM_initialize_doc},
	{"train", (PyCFunction)PatchMCBM_train, METH_KEYWORDS, PatchMCBM_train_doc},
	{"preconditioner", (PyCFunction)PatchMCBM_preconditioner, METH_VARARGS, 0},
	{"__reduce__", (PyCFunction)PatchMCBM_reduce, METH_NOARGS, PatchMCBM_reduce_doc},
	{"__setstate__", (PyCFunction)PatchMCBM_setstate, METH_VARARGS, PatchMCBM_setstate_doc},
	{0}
};

PyTypeObject PatchMCBM_type = {
	PyObject_HEAD_INIT(0)
	0,                                /*ob_size*/
	"cmt.models.PatchMCBM",           /*tp_name*/
	sizeof(PatchMCBMObject),          /*tp_basicsize*/
	0,                                /*tp_itemsize*/
	(destructor)Distribution_dealloc, /*tp_dealloc*/
	0,                                /*tp_print*/
	0,                                /*tp_getattr*/
	0,                                /*tp_setattr*/
	0,                                /*tp_compare*/
	0,                                /*tp_repr*/
	0,                                /*tp_as_number*/
	0,                                /*tp_as_sequence*/
	&PatchMCBM_as_mapping,            /*tp_as_mapping*/
	0,                                /*tp_hash */
	0,                                /*tp_call*/
	0,                                /*tp_str*/
	0,                                /*tp_getattro*/
	0,                                /*tp_setattro*/
	0,                                /*tp_as_buffer*/
	Py_TPFLAGS_DEFAULT,               /*tp_flags*/
	PatchMCBM_doc,                    /*tp_doc*/
	0,                                /*tp_traverse*/
	0,                                /*tp_clear*/
	0,                                /*tp_richcompare*/
	0,                                /*tp_weaklistoffset*/
	0,                                /*tp_iter*/
	0,                                /*tp_iternext*/
	PatchMCBM_methods,                /*tp_methods*/
	0,                                /*tp_members*/
	PatchMCBM_getset,                 /*tp_getset*/
	&PatchModel_type,                 /*tp_base*/
	0,                                /*tp_dict*/
	0,                                /*tp_descr_get*/
	0,                                /*tp_descr_set*/
	0,                                /*tp_dictoffset*/
	(initproc)PatchMCBM_init,         /*tp_init*/
	0,                                /*tp_alloc*/
	Distribution_new,                 /*tp_new*/
};

static PyMappingMethods PatchMCGSM_as_mapping = {
	0,                                      /*mp_length*/
	(binaryfunc)PatchMCGSM_subscript,        /*mp_subscript*/
	(objobjargproc)PatchMCGSM_ass_subscript, /*mp_ass_subscript*/
};

static PyGetSetDef PatchMCGSM_getset[] = {
	{"preconditioners", 
		(getter)PatchMCGSM_preconditioners,
		(setter)PatchMCGSM_set_preconditioners,
		"A dictionary containing all preconditioners."},
	{0}
};

static PyMethodDef PatchMCGSM_methods[] = {
	{"initialize", (PyCFunction)PatchMCGSM_initialize, METH_KEYWORDS, PatchMCGSM_initialize_doc},
	{"train", (PyCFunction)PatchMCGSM_train, METH_KEYWORDS, PatchMCGSM_train_doc},
	{"preconditioner", (PyCFunction)PatchMCGSM_preconditioner, METH_VARARGS, 0},
	{"__reduce__", (PyCFunction)PatchMCGSM_reduce, METH_NOARGS, PatchMCGSM_reduce_doc},
	{"__setstate__", (PyCFunction)PatchMCGSM_setstate, METH_VARARGS, PatchMCGSM_setstate_doc},
	{0}
};

PyTypeObject PatchMCGSM_type = {
	PyObject_HEAD_INIT(0)
	0,                                /*ob_size*/
	"cmt.models.PatchMCGSM",          /*tp_name*/
	sizeof(PatchMCGSMObject),         /*tp_basicsize*/
	0,                                /*tp_itemsize*/
	(destructor)Distribution_dealloc, /*tp_dealloc*/
	0,                                /*tp_print*/
	0,                                /*tp_getattr*/
	0,                                /*tp_setattr*/
	0,                                /*tp_compare*/
	0,                                /*tp_repr*/
	0,                                /*tp_as_number*/
	0,                                /*tp_as_sequence*/
	&PatchMCGSM_as_mapping,           /*tp_as_mapping*/
	0,                                /*tp_hash */
	0,                                /*tp_call*/
	0,                                /*tp_str*/
	0,                                /*tp_getattro*/
	0,                                /*tp_setattro*/
	0,                                /*tp_as_buffer*/
	Py_TPFLAGS_DEFAULT,               /*tp_flags*/
	PatchMCGSM_doc,                   /*tp_doc*/
	0,                                /*tp_traverse*/
	0,                                /*tp_clear*/
	0,                                /*tp_richcompare*/
	0,                                /*tp_weaklistoffset*/
	0,                                /*tp_iter*/
	0,                                /*tp_iternext*/
	PatchMCGSM_methods,               /*tp_methods*/
	0,                                /*tp_members*/
	PatchMCGSM_getset,                /*tp_getset*/
	&PatchModel_type,                 /*tp_base*/
	0,                                /*tp_dict*/
	0,                                /*tp_descr_get*/
	0,                                /*tp_descr_set*/
	0,                                /*tp_dictoffset*/
	(initproc)PatchMCGSM_init,        /*tp_init*/
	0,                                /*tp_alloc*/
	Distribution_new,                 /*tp_new*/
};

static PyGetSetDef GLM_getset[] = {
	{"weights",
		(getter)GLM_weights,
		(setter)GLM_set_weights,
		"Linear filter, $\\mathbf{w}$."},
	{"bias",
		(getter)GLM_bias,
		(setter)GLM_set_bias,
		"Bias term, $b$."},
	{"nonlinearity",
		(getter)GLM_nonlinearity,
		(setter)GLM_set_nonlinearity,
		"Nonlinearity applied to output of linear filter, $g$."},
	{"distribution",
		(getter)GLM_distribution,
		(setter)GLM_set_distribution,
		"Distribution whose average value is determined by output of nonlinearity."},
	{0}
};

static PyMethodDef GLM_methods[] = {
	{"train", (PyCFunction)GLM_train, METH_VARARGS | METH_KEYWORDS, GLM_train_doc},
	{"_parameters",
		(PyCFunction)GLM_parameters,
		METH_VARARGS | METH_KEYWORDS,
		Trainable_parameters_doc},
	{"_set_parameters",
		(PyCFunction)GLM_set_parameters,
		METH_VARARGS | METH_KEYWORDS,
		Trainable_set_parameters_doc},
	{"_parameter_gradient",
		(PyCFunction)GLM_parameter_gradient,
		METH_VARARGS | METH_KEYWORDS, 
		Trainable_parameter_gradient_doc},
	{"_fisher_information",
		(PyCFunction)GLM_fisher_information,
		METH_VARARGS | METH_KEYWORDS, 
		Trainable_fisher_information_doc},
	{"_check_gradient",
		(PyCFunction)GLM_check_gradient,
		METH_VARARGS | METH_KEYWORDS,
		Trainable_check_gradient_doc},
	{"_check_performance",
		(PyCFunction)GLM_check_performance,
		METH_VARARGS | METH_KEYWORDS,
		Trainable_check_performance_doc},
	{"__reduce__", (PyCFunction)GLM_reduce, METH_NOARGS, GLM_reduce_doc},
	{"__setstate__", (PyCFunction)GLM_setstate, METH_VARARGS, GLM_setstate_doc},
	{0}
};

PyTypeObject GLM_type = {
	PyObject_HEAD_INIT(0)
	0,                       /*ob_size*/
	"cmt.models.GLM",        /*tp_name*/
	sizeof(GLMObject),       /*tp_basicsize*/
	0,                       /*tp_itemsize*/
	(destructor)GLM_dealloc, /*tp_dealloc*/
	0,                       /*tp_print*/
	0,                       /*tp_getattr*/
	0,                       /*tp_setattr*/
	0,                       /*tp_compare*/
	0,                       /*tp_repr*/
	0,                       /*tp_as_number*/
	0,                       /*tp_as_sequence*/
	0,                       /*tp_as_mapping*/
	0,                       /*tp_hash */
	0,                       /*tp_call*/
	0,                       /*tp_str*/
	0,                       /*tp_getattro*/
	0,                       /*tp_setattro*/
	0,                       /*tp_as_buffer*/
	Py_TPFLAGS_DEFAULT,      /*tp_flags*/
	GLM_doc,                 /*tp_doc*/
	0,                       /*tp_traverse*/
	0,                       /*tp_clear*/
	0,                       /*tp_richcompare*/
	0,                       /*tp_weaklistoffset*/
	0,                       /*tp_iter*/
	0,                       /*tp_iternext*/
	GLM_methods,             /*tp_methods*/
	0,                       /*tp_members*/
	GLM_getset,              /*tp_getset*/
	&CD_type,                /*tp_base*/
	0,                       /*tp_dict*/
	0,                       /*tp_descr_get*/
	0,                       /*tp_descr_set*/
	0,                       /*tp_dictoffset*/
	(initproc)GLM_init,      /*tp_init*/
	0,                       /*tp_alloc*/
	CD_new,                  /*tp_new*/
};

static PyGetSetDef MLR_getset[] = {
	{"weights",
		(getter)MLR_weights,
		(setter)MLR_set_weights,
		"Linear filters, $\\mathbf{w}_i$, one per row."},
	{"biases",
		(getter)MLR_biases,
		(setter)MLR_set_biases,
		"Bias terms, $b_i$."},
	{0}
};

static PyMethodDef MLR_methods[] = {
	{"train", (PyCFunction)MLR_train, METH_VARARGS | METH_KEYWORDS, MLR_train_doc},
	{"_parameters",
		(PyCFunction)MLR_parameters,
		METH_VARARGS | METH_KEYWORDS,
		Trainable_parameters_doc},
	{"_set_parameters",
		(PyCFunction)MLR_set_parameters,
		METH_VARARGS | METH_KEYWORDS,
		Trainable_set_parameters_doc},
	{"_parameter_gradient",
		(PyCFunction)MLR_parameter_gradient,
		METH_VARARGS | METH_KEYWORDS, 0},
	{"_check_gradient",
		(PyCFunction)MLR_check_gradient,
		METH_VARARGS | METH_KEYWORDS,
		Trainable_check_gradient_doc},
	{"_check_performance",
		(PyCFunction)STM_check_performance,
		METH_VARARGS | METH_KEYWORDS,
		Trainable_check_performance_doc},
	{"__reduce__", (PyCFunction)MLR_reduce, METH_NOARGS, MLR_reduce_doc},
	{"__setstate__", (PyCFunction)MLR_setstate, METH_VARARGS, MLR_setstate_doc},
	{0}
};

PyTypeObject MLR_type = {
	PyObject_HEAD_INIT(0)
	0,                       /*ob_size*/
	"cmt.models.MLR",        /*tp_name*/
	sizeof(MLRObject),       /*tp_basicsize*/
	0,                       /*tp_itemsize*/
	(destructor)MLR_dealloc, /*tp_dealloc*/
	0,                       /*tp_print*/
	0,                       /*tp_getattr*/
	0,                       /*tp_setattr*/
	0,                       /*tp_compare*/
	0,                       /*tp_repr*/
	0,                       /*tp_as_number*/
	0,                       /*tp_as_sequence*/
	0,                       /*tp_as_mapping*/
	0,                       /*tp_hash */
	0,                       /*tp_call*/
	0,                       /*tp_str*/
	0,                       /*tp_getattro*/
	0,                       /*tp_setattro*/
	0,                       /*tp_as_buffer*/
	Py_TPFLAGS_DEFAULT,      /*tp_flags*/
	MLR_doc,                 /*tp_doc*/
	0,                       /*tp_traverse*/
	0,                       /*tp_clear*/
	0,                       /*tp_richcompare*/
	0,                       /*tp_weaklistoffset*/
	0,                       /*tp_iter*/
	0,                       /*tp_iternext*/
	MLR_methods,             /*tp_methods*/
	0,                       /*tp_members*/
	MLR_getset,              /*tp_getset*/
	&CD_type,                /*tp_base*/
	0,                       /*tp_dict*/
	0,                       /*tp_descr_get*/
	0,                       /*tp_descr_set*/
	0,                       /*tp_dictoffset*/
	(initproc)MLR_init,      /*tp_init*/
	0,                       /*tp_alloc*/
	CD_new,                  /*tp_new*/
};

static PyMappingMethods FVBN_as_mapping = {
	0,                                      /*mp_length*/
	(binaryfunc)FVBN_subscript,        /*mp_subscript*/
	(objobjargproc)FVBN_ass_subscript, /*mp_ass_subscript*/
};

static PyGetSetDef FVBN_getset[] = {
	{"preconditioners", 
		(getter)FVBN_preconditioners,
		(setter)FVBN_set_preconditioners,
		"A dictionary containing all preconditioners."},
	{0}
};

static PyMethodDef FVBN_methods[] = {
	{"initialize", (PyCFunction)FVBN_initialize, METH_KEYWORDS, FVBN_initialize_doc},
	{"train", (PyCFunction)FVBN_train, METH_KEYWORDS, FVBN_train_doc},
	{"preconditioner", (PyCFunction)FVBN_preconditioner, METH_VARARGS, 0},
	{"__reduce__", (PyCFunction)FVBN_reduce, METH_NOARGS, FVBN_reduce_doc},
	{"__setstate__", (PyCFunction)FVBN_setstate, METH_VARARGS, FVBN_setstate_doc},
	{0}
};

PyTypeObject FVBN_type = {
	PyObject_HEAD_INIT(0)
	0,                                /*ob_size*/
	"cmt.models.FVBN",                /*tp_name*/
	sizeof(FVBNObject),               /*tp_basicsize*/
	0,                                /*tp_itemsize*/
	(destructor)Distribution_dealloc, /*tp_dealloc*/
	0,                                /*tp_print*/
	0,                                /*tp_getattr*/
	0,                                /*tp_setattr*/
	0,                                /*tp_compare*/
	0,                                /*tp_repr*/
	0,                                /*tp_as_number*/
	0,                                /*tp_as_sequence*/
	&FVBN_as_mapping,                 /*tp_as_mapping*/
	0,                                /*tp_hash */
	0,                                /*tp_call*/
	0,                                /*tp_str*/
	0,                                /*tp_getattro*/
	0,                                /*tp_setattro*/
	0,                                /*tp_as_buffer*/
	Py_TPFLAGS_DEFAULT,               /*tp_flags*/
	FVBN_doc,                         /*tp_doc*/
	0,                                /*tp_traverse*/
	0,                                /*tp_clear*/
	0,                                /*tp_richcompare*/
	0,                                /*tp_weaklistoffset*/
	0,                                /*tp_iter*/
	0,                                /*tp_iternext*/
	FVBN_methods,                     /*tp_methods*/
	0,                                /*tp_members*/
	FVBN_getset,                      /*tp_getset*/
	&PatchModel_type,                 /*tp_base*/
	0,                                /*tp_dict*/
	0,                                /*tp_descr_get*/
	0,                                /*tp_descr_set*/
	0,                                /*tp_dictoffset*/
	(initproc)FVBN_init,              /*tp_init*/
	0,                                /*tp_alloc*/
	Distribution_new,                 /*tp_new*/
};

static PyMethodDef Nonlinearity_methods[] = {
	{"__reduce__", (PyCFunction)Nonlinearity_reduce, METH_NOARGS, Nonlinearity_reduce_doc},
	{0}
};

PyTypeObject Nonlinearity_type = {
	PyObject_HEAD_INIT(0)
	0,                                /*ob_size*/
	"cmt.nonlinear.Nonlinearity",     /*tp_name*/
	sizeof(NonlinearityObject),       /*tp_basicsize*/
	0,                                /*tp_itemsize*/
	(destructor)Nonlinearity_dealloc, /*tp_dealloc*/
	0,                                /*tp_print*/
	0,                                /*tp_getattr*/
	0,                                /*tp_setattr*/
	0,                                /*tp_compare*/
	0,                                /*tp_repr*/
	0,                                /*tp_as_number*/
	0,                                /*tp_as_sequence*/
	0,                                /*tp_as_mapping*/
	0,                                /*tp_hash */
	(ternaryfunc)Nonlinearity_call,   /*tp_call*/
	0,                                /*tp_str*/
	0,                                /*tp_getattro*/
	0,                                /*tp_setattro*/
	0,                                /*tp_as_buffer*/
	Py_TPFLAGS_DEFAULT,               /*tp_flags*/
	Nonlinearity_doc,                 /*tp_doc*/
	0,                                /*tp_traverse*/
	0,                                /*tp_clear*/
	0,                                /*tp_richcompare*/
	0,                                /*tp_weaklistoffset*/
	0,                                /*tp_iter*/
	0,                                /*tp_iternext*/
	Nonlinearity_methods,             /*tp_methods*/
	0,                                /*tp_members*/
	0,                                /*tp_getset*/
	0,                                /*tp_base*/
	0,                                /*tp_dict*/
	0,                                /*tp_descr_get*/
	0,                                /*tp_descr_set*/
	0,                                /*tp_dictoffset*/
	(initproc)Nonlinearity_init,      /*tp_init*/
	0,                                /*tp_alloc*/
	Nonlinearity_new,                 /*tp_new*/
};

PyTypeObject DifferentiableNonlinearity_type = {
	PyObject_HEAD_INIT(0)
	0,                                          /*ob_size*/
	"cmt.nonlinear.DifferentiableNonlinearity", /*tp_name*/
	sizeof(DifferentiableNonlinearityObject),   /*tp_basicsize*/
	0,                                          /*tp_itemsize*/
	(destructor)Nonlinearity_dealloc,           /*tp_dealloc*/
	0,                                          /*tp_print*/
	0,                                          /*tp_getattr*/
	0,                                          /*tp_setattr*/
	0,                                          /*tp_compare*/
	0,                                          /*tp_repr*/
	0,                                          /*tp_as_number*/
	0,                                          /*tp_as_sequence*/
	0,                                          /*tp_as_mapping*/
	0,                                          /*tp_hash */
	0,                                          /*tp_call*/
	0,                                          /*tp_str*/
	0,                                          /*tp_getattro*/
	0,                                          /*tp_setattro*/
	0,                                          /*tp_as_buffer*/
	Py_TPFLAGS_DEFAULT,                         /*tp_flags*/
	Nonlinearity_doc,                           /*tp_doc*/
	0,                                          /*tp_traverse*/
	0,                                          /*tp_clear*/
	0,                                          /*tp_richcompare*/
	0,                                          /*tp_weaklistoffset*/
	0,                                          /*tp_iter*/
	0,                                          /*tp_iternext*/
	0,                                          /*tp_methods*/
	0,                                          /*tp_members*/
	0,                                          /*tp_getset*/
	&Nonlinearity_type,                         /*tp_base*/
	0,                                          /*tp_dict*/
	0,                                          /*tp_descr_get*/
	0,                                          /*tp_descr_set*/
	0,                                          /*tp_dictoffset*/
	(initproc)Nonlinearity_init,                /*tp_init*/
	0,                                          /*tp_alloc*/
	Nonlinearity_new,                           /*tp_new*/
};

PyTypeObject InvertibleNonlinearity_type = {
	PyObject_HEAD_INIT(0)
	0,                                      /*ob_size*/
	"cmt.nonlinear.InvertibleNonlinearity", /*tp_name*/
	sizeof(InvertibleNonlinearityObject),   /*tp_basicsize*/
	0,                                      /*tp_itemsize*/
	(destructor)Nonlinearity_dealloc,       /*tp_dealloc*/
	0,                                      /*tp_print*/
	0,                                      /*tp_getattr*/
	0,                                      /*tp_setattr*/
	0,                                      /*tp_compare*/
	0,                                      /*tp_repr*/
	0,                                      /*tp_as_number*/
	0,                                      /*tp_as_sequence*/
	0,                                      /*tp_as_mapping*/
	0,                                      /*tp_hash */
	0,                                      /*tp_call*/
	0,                                      /*tp_str*/
	0,                                      /*tp_getattro*/
	0,                                      /*tp_setattro*/
	0,                                      /*tp_as_buffer*/
	Py_TPFLAGS_DEFAULT,                     /*tp_flags*/
	Nonlinearity_doc,                       /*tp_doc*/
	0,                                      /*tp_traverse*/
	0,                                      /*tp_clear*/
	0,                                      /*tp_richcompare*/
	0,                                      /*tp_weaklistoffset*/
	0,                                      /*tp_iter*/
	0,                                      /*tp_iternext*/
	0,                                      /*tp_methods*/
	0,                                      /*tp_members*/
	0,                                      /*tp_getset*/
	&Nonlinearity_type,                     /*tp_base*/
	0,                                      /*tp_dict*/
	0,                                      /*tp_descr_get*/
	0,                                      /*tp_descr_set*/
	0,                                      /*tp_dictoffset*/
	(initproc)Nonlinearity_init,            /*tp_init*/
	0,                                      /*tp_alloc*/
	Nonlinearity_new,                       /*tp_new*/
};

PyTypeObject TrainableNonlinearity_type = {
	PyObject_HEAD_INIT(0)
	0,                                     /*ob_size*/
	"cmt.nonlinear.TrainableNonlinearity", /*tp_name*/
	sizeof(TrainableNonlinearityObject),   /*tp_basicsize*/
	0,                                     /*tp_itemsize*/
	(destructor)Nonlinearity_dealloc,      /*tp_dealloc*/
	0,                                     /*tp_print*/
	0,                                     /*tp_getattr*/
	0,                                     /*tp_setattr*/
	0,                                     /*tp_compare*/
	0,                                     /*tp_repr*/
	0,                                     /*tp_as_number*/
	0,                                     /*tp_as_sequence*/
	0,                                     /*tp_as_mapping*/
	0,                                     /*tp_hash */
	0,                                     /*tp_call*/
	0,                                     /*tp_str*/
	0,                                     /*tp_getattro*/
	0,                                     /*tp_setattro*/
	0,                                     /*tp_as_buffer*/
	Py_TPFLAGS_DEFAULT,                    /*tp_flags*/
	Nonlinearity_doc,                      /*tp_doc*/
	0,                                     /*tp_traverse*/
	0,                                     /*tp_clear*/
	0,                                     /*tp_richcompare*/
	0,                                     /*tp_weaklistoffset*/
	0,                                     /*tp_iter*/
	0,                                     /*tp_iternext*/
	0,                                     /*tp_methods*/
	0,                                     /*tp_members*/
	0,                                     /*tp_getset*/
	&Nonlinearity_type,                    /*tp_base*/
	0,                                     /*tp_dict*/
	0,                                     /*tp_descr_get*/
	0,                                     /*tp_descr_set*/
	0,                                     /*tp_dictoffset*/
	(initproc)Nonlinearity_init,           /*tp_init*/
	0,                                     /*tp_alloc*/
	Nonlinearity_new,                      /*tp_new*/
};

PyTypeObject LogisticFunction_type = {
	PyObject_HEAD_INIT(0)
	0,                                /*ob_size*/
	"cmt.nonlinear.LogisticFunction", /*tp_name*/
	sizeof(LogisticFunctionObject),   /*tp_basicsize*/
	0,                                /*tp_itemsize*/
	(destructor)Nonlinearity_dealloc, /*tp_dealloc*/
	0,                                /*tp_print*/
	0,                                /*tp_getattr*/
	0,                                /*tp_setattr*/
	0,                                /*tp_compare*/
	0,                                /*tp_repr*/
	0,                                /*tp_as_number*/
	0,                                /*tp_as_sequence*/
	0,                                /*tp_as_mapping*/
	0,                                /*tp_hash */
	0,                                /*tp_call*/
	0,                                /*tp_str*/
	0,                                /*tp_getattro*/
	0,                                /*tp_setattro*/
	0,                                /*tp_as_buffer*/
	Py_TPFLAGS_DEFAULT,               /*tp_flags*/
	LogisticFunction_doc,             /*tp_doc*/
	0,                                /*tp_traverse*/
	0,                                /*tp_clear*/
	0,                                /*tp_richcompare*/
	0,                                /*tp_weaklistoffset*/
	0,                                /*tp_iter*/
	0,                                /*tp_iternext*/
	0,                                /*tp_methods*/
	0,                                /*tp_members*/
	0,                                /*tp_getset*/
	&DifferentiableNonlinearity_type, /*tp_base*/
	0,                                /*tp_dict*/
	0,                                /*tp_descr_get*/
	0,                                /*tp_descr_set*/
	0,                                /*tp_dictoffset*/
	(initproc)LogisticFunction_init,  /*tp_init*/
	0,                                /*tp_alloc*/
	Nonlinearity_new,                 /*tp_new*/
};

PyTypeObject ExponentialFunction_type = {
	PyObject_HEAD_INIT(0)
	0,                                   /*ob_size*/
	"cmt.nonlinear.ExponentialFunction", /*tp_name*/
	sizeof(ExponentialFunctionObject),   /*tp_basicsize*/
	0,                                   /*tp_itemsize*/
	(destructor)Nonlinearity_dealloc,    /*tp_dealloc*/
	0,                                   /*tp_print*/
	0,                                   /*tp_getattr*/
	0,                                   /*tp_setattr*/
	0,                                   /*tp_compare*/
	0,                                   /*tp_repr*/
	0,                                   /*tp_as_number*/
	0,                                   /*tp_as_sequence*/
	0,                                   /*tp_as_mapping*/
	0,                                   /*tp_hash */
	0,                                   /*tp_call*/
	0,                                   /*tp_str*/
	0,                                   /*tp_getattro*/
	0,                                   /*tp_setattro*/
	0,                                   /*tp_as_buffer*/
	Py_TPFLAGS_DEFAULT,                  /*tp_flags*/
	ExponentialFunction_doc,             /*tp_doc*/
	0,                                   /*tp_traverse*/
	0,                                   /*tp_clear*/
	0,                                   /*tp_richcompare*/
	0,                                   /*tp_weaklistoffset*/
	0,                                   /*tp_iter*/
	0,                                   /*tp_iternext*/
	0,                                   /*tp_methods*/
	0,                                   /*tp_members*/
	0,                                   /*tp_getset*/
	&DifferentiableNonlinearity_type,    /*tp_base*/
	0,                                   /*tp_dict*/
	0,                                   /*tp_descr_get*/
	0,                                   /*tp_descr_set*/
	0,                                   /*tp_dictoffset*/
	(initproc)ExponentialFunction_init,  /*tp_init*/
	0,                                   /*tp_alloc*/
	Nonlinearity_new,                    /*tp_new*/
};

static PyMethodDef HistogramNonlinearity_methods[] = {
	{"__reduce__", (PyCFunction)HistogramNonlinearity_reduce, METH_NOARGS, Nonlinearity_reduce_doc},
	{"__setstate__", (PyCFunction)HistogramNonlinearity_setstate, METH_VARARGS, 0},
	{0}
};

PyTypeObject HistogramNonlinearity_type = {
	PyObject_HEAD_INIT(0)
	0,                                     /*ob_size*/
	"cmt.nonlinear.HistogramNonlinearity", /*tp_name*/
	sizeof(HistogramNonlinearityObject),   /*tp_basicsize*/
	0,                                     /*tp_itemsize*/
	(destructor)Nonlinearity_dealloc,      /*tp_dealloc*/
	0,                                     /*tp_print*/
	0,                                     /*tp_getattr*/
	0,                                     /*tp_setattr*/
	0,                                     /*tp_compare*/
	0,                                     /*tp_repr*/
	0,                                     /*tp_as_number*/
	0,                                     /*tp_as_sequence*/
	0,                                     /*tp_as_mapping*/
	0,                                     /*tp_hash */
	0,                                     /*tp_call*/
	0,                                     /*tp_str*/
	0,                                     /*tp_getattro*/
	0,                                     /*tp_setattro*/
	0,                                     /*tp_as_buffer*/
	Py_TPFLAGS_DEFAULT,                    /*tp_flags*/
	HistogramNonlinearity_doc,             /*tp_doc*/
	0,                                     /*tp_traverse*/
	0,                                     /*tp_clear*/
	0,                                     /*tp_richcompare*/
	0,                                     /*tp_weaklistoffset*/
	0,                                     /*tp_iter*/
	0,                                     /*tp_iternext*/
	HistogramNonlinearity_methods,         /*tp_methods*/
	0,                                     /*tp_members*/
	0,                                     /*tp_getset*/
	&TrainableNonlinearity_type,           /*tp_base*/
	0,                                     /*tp_dict*/
	0,                                     /*tp_descr_get*/
	0,                                     /*tp_descr_set*/
	0,                                     /*tp_dictoffset*/
	(initproc)HistogramNonlinearity_init,  /*tp_init*/
	0,                                     /*tp_alloc*/
	Nonlinearity_new,                      /*tp_new*/
};

static PyMethodDef BlobNonlinearity_methods[] = {
	{"__reduce__", (PyCFunction)BlobNonlinearity_reduce, METH_NOARGS, Nonlinearity_reduce_doc},
	{"__setstate__", (PyCFunction)BlobNonlinearity_setstate, METH_VARARGS, 0},
	{0}
};

PyTypeObject BlobNonlinearity_type = {
	PyObject_HEAD_INIT(0)
	0,                                /*ob_size*/
	"cmt.nonlinear.BlobNonlinearity", /*tp_name*/
	sizeof(BlobNonlinearityObject),   /*tp_basicsize*/
	0,                                /*tp_itemsize*/
	(destructor)Nonlinearity_dealloc, /*tp_dealloc*/
	0,                                /*tp_print*/
	0,                                /*tp_getattr*/
	0,                                /*tp_setattr*/
	0,                                /*tp_compare*/
	0,                                /*tp_repr*/
	0,                                /*tp_as_number*/
	0,                                /*tp_as_sequence*/
	0,                                /*tp_as_mapping*/
	0,                                /*tp_hash */
	0,                                /*tp_call*/
	0,                                /*tp_str*/
	0,                                /*tp_getattro*/
	0,                                /*tp_setattro*/
	0,                                /*tp_as_buffer*/
	Py_TPFLAGS_DEFAULT,               /*tp_flags*/
	BlobNonlinearity_doc,             /*tp_doc*/
	0,                                /*tp_traverse*/
	0,                                /*tp_clear*/
	0,                                /*tp_richcompare*/
	0,                                /*tp_weaklistoffset*/
	0,                                /*tp_iter*/
	0,                                /*tp_iternext*/
	BlobNonlinearity_methods,         /*tp_methods*/
	0,                                /*tp_members*/
	0,                                /*tp_getset*/
	&TrainableNonlinearity_type,      /*tp_base*/
	0,                                /*tp_dict*/
	0,                                /*tp_descr_get*/
	0,                                /*tp_descr_set*/
	0,                                /*tp_dictoffset*/
	(initproc)BlobNonlinearity_init,  /*tp_init*/
	0,                                /*tp_alloc*/
	Nonlinearity_new,                 /*tp_new*/
};

static PyMethodDef TanhBlobNonlinearity_methods[] = {
	{"__reduce__", (PyCFunction)TanhBlobNonlinearity_reduce, METH_NOARGS, Nonlinearity_reduce_doc},
	{"__setstate__", (PyCFunction)TanhBlobNonlinearity_setstate, METH_VARARGS, 0},
	{0}
};

PyTypeObject TanhBlobNonlinearity_type = {
	PyObject_HEAD_INIT(0)
	0,                                    /*ob_size*/
	"cmt.nonlinear.TanhBlobNonlinearity", /*tp_name*/
	sizeof(TanhBlobNonlinearityObject),   /*tp_basicsize*/
	0,                                    /*tp_itemsize*/
	(destructor)Nonlinearity_dealloc,     /*tp_dealloc*/
	0,                                    /*tp_print*/
	0,                                    /*tp_getattr*/
	0,                                    /*tp_setattr*/
	0,                                    /*tp_compare*/
	0,                                    /*tp_repr*/
	0,                                    /*tp_as_number*/
	0,                                    /*tp_as_sequence*/
	0,                                    /*tp_as_mapping*/
	0,                                    /*tp_hash */
	0,                                    /*tp_call*/
	0,                                    /*tp_str*/
	0,                                    /*tp_getattro*/
	0,                                    /*tp_setattro*/
	0,                                    /*tp_as_buffer*/
	Py_TPFLAGS_DEFAULT,                   /*tp_flags*/
	TanhBlobNonlinearity_doc,             /*tp_doc*/
	0,                                    /*tp_traverse*/
	0,                                    /*tp_clear*/
	0,                                    /*tp_richcompare*/
	0,                                    /*tp_weaklistoffset*/
	0,                                    /*tp_iter*/
	0,                                    /*tp_iternext*/
	TanhBlobNonlinearity_methods,         /*tp_methods*/
	0,                                    /*tp_members*/
	0,                                    /*tp_getset*/
	&TrainableNonlinearity_type,          /*tp_base*/
	0,                                    /*tp_dict*/
	0,                                    /*tp_descr_get*/
	0,                                    /*tp_descr_set*/
	0,                                    /*tp_dictoffset*/
	(initproc)TanhBlobNonlinearity_init,  /*tp_init*/
	0,                                    /*tp_alloc*/
	Nonlinearity_new,                     /*tp_new*/
};

PyTypeObject UnivariateDistribution_type = {
	PyObject_HEAD_INIT(0)
	0,                                     /*ob_size*/
	"cmt.models.UnivariateDistribution",   /*tp_name*/
	sizeof(UnivariateDistributionObject),  /*tp_basicsize*/
	0,                                     /*tp_itemsize*/
	(destructor)Distribution_dealloc,      /*tp_dealloc*/
	0,                                     /*tp_print*/
	0,                                     /*tp_getattr*/
	0,                                     /*tp_setattr*/
	0,                                     /*tp_compare*/
	0,                                     /*tp_repr*/
	0,                                     /*tp_as_number*/
	0,                                     /*tp_as_sequence*/
	0,                                     /*tp_as_mapping*/
	0,                                     /*tp_hash */
	0,                                     /*tp_call*/
	0,                                     /*tp_str*/
	0,                                     /*tp_getattro*/
	0,                                     /*tp_setattro*/
	0,                                     /*tp_as_buffer*/
	Py_TPFLAGS_DEFAULT,                    /*tp_flags*/
	UnivariateDistribution_doc,            /*tp_doc*/
	0,                                     /*tp_traverse*/
	0,                                     /*tp_clear*/
	0,                                     /*tp_richcompare*/
	0,                                     /*tp_weaklistoffset*/
	0,                                     /*tp_iter*/
	0,                                     /*tp_iternext*/
	0,                                     /*tp_methods*/
	0,                                     /*tp_members*/
	0,                                     /*tp_getset*/
	&Distribution_type,                    /*tp_base*/
	0,                                     /*tp_dict*/
	0,                                     /*tp_descr_get*/
	0,                                     /*tp_descr_set*/
	0,                                     /*tp_dictoffset*/
	(initproc)UnivariateDistribution_init, /*tp_init*/
	0,                                     /*tp_alloc*/
	Distribution_new,                      /*tp_new*/
};

static PyMethodDef Bernoulli_methods[] = {
	{"__reduce__", (PyCFunction)Bernoulli_reduce, METH_NOARGS, Bernoulli_reduce_doc},
	{0}
};

PyTypeObject Bernoulli_type = {
	PyObject_HEAD_INIT(0)
	0,                                /*ob_size*/
	"cmt.models.Bernoulli",           /*tp_name*/
	sizeof(BernoulliObject),          /*tp_basicsize*/
	0,                                /*tp_itemsize*/
	(destructor)Distribution_dealloc, /*tp_dealloc*/
	0,                                /*tp_print*/
	0,                                /*tp_getattr*/
	0,                                /*tp_setattr*/
	0,                                /*tp_compare*/
	0,                                /*tp_repr*/
	0,                                /*tp_as_number*/
	0,                                /*tp_as_sequence*/
	0,                                /*tp_as_mapping*/
	0,                                /*tp_hash */
	0,                                /*tp_call*/
	0,                                /*tp_str*/
	0,                                /*tp_getattro*/
	0,                                /*tp_setattro*/
	0,                                /*tp_as_buffer*/
	Py_TPFLAGS_DEFAULT,               /*tp_flags*/
	Bernoulli_doc,                    /*tp_doc*/
	0,                                /*tp_traverse*/
	0,                                /*tp_clear*/
	0,                                /*tp_richcompare*/
	0,                                /*tp_weaklistoffset*/
	0,                                /*tp_iter*/
	0,                                /*tp_iternext*/
	Bernoulli_methods,                /*tp_methods*/
	0,                                /*tp_members*/
	0,                                /*tp_getset*/
	&UnivariateDistribution_type,     /*tp_base*/
	0,                                /*tp_dict*/
	0,                                /*tp_descr_get*/
	0,                                /*tp_descr_set*/
	0,                                /*tp_dictoffset*/
	(initproc)Bernoulli_init,         /*tp_init*/
	0,                                /*tp_alloc*/
	Distribution_new,                 /*tp_new*/
};

static PyMethodDef Poisson_methods[] = {
	{"__reduce__", (PyCFunction)Poisson_reduce, METH_NOARGS, Poisson_reduce_doc},
	{0}
};

PyTypeObject Poisson_type = {
	PyObject_HEAD_INIT(0)
	0,                                /*ob_size*/
	"cmt.models.Poisson",             /*tp_name*/
	sizeof(PoissonObject),            /*tp_basicsize*/
	0,                                /*tp_itemsize*/
	(destructor)Distribution_dealloc, /*tp_dealloc*/
	0,                                /*tp_print*/
	0,                                /*tp_getattr*/
	0,                                /*tp_setattr*/
	0,                                /*tp_compare*/
	0,                                /*tp_repr*/
	0,                                /*tp_as_number*/
	0,                                /*tp_as_sequence*/
	0,                                /*tp_as_mapping*/
	0,                                /*tp_hash */
	0,                                /*tp_call*/
	0,                                /*tp_str*/
	0,                                /*tp_getattro*/
	0,                                /*tp_setattro*/
	0,                                /*tp_as_buffer*/
	Py_TPFLAGS_DEFAULT,               /*tp_flags*/
	Poisson_doc,                      /*tp_doc*/
	0,                                /*tp_traverse*/
	0,                                /*tp_clear*/
	0,                                /*tp_richcompare*/
	0,                                /*tp_weaklistoffset*/
	0,                                /*tp_iter*/
	0,                                /*tp_iternext*/
	Poisson_methods,                  /*tp_methods*/
	0,                                /*tp_members*/
	0,                                /*tp_getset*/
	&UnivariateDistribution_type,     /*tp_base*/
	0,                                /*tp_dict*/
	0,                                /*tp_descr_get*/
	0,                                /*tp_descr_set*/
	0,                                /*tp_dictoffset*/
	(initproc)Poisson_init,           /*tp_init*/
	0,                                /*tp_alloc*/
	Distribution_new,                 /*tp_new*/
};

static PyMethodDef Binomial_methods[] = {
	{"__reduce__", (PyCFunction)Binomial_reduce, METH_NOARGS, Binomial_reduce_doc},
	{0}
};

PyTypeObject Binomial_type = {
	PyObject_HEAD_INIT(0)
	0,                                /*ob_size*/
	"cmt.models.Binomial",            /*tp_name*/
	sizeof(BinomialObject),           /*tp_basicsize*/
	0,                                /*tp_itemsize*/
	(destructor)Distribution_dealloc, /*tp_dealloc*/
	0,                                /*tp_print*/
	0,                                /*tp_getattr*/
	0,                                /*tp_setattr*/
	0,                                /*tp_compare*/
	0,                                /*tp_repr*/
	0,                                /*tp_as_number*/
	0,                                /*tp_as_sequence*/
	0,                                /*tp_as_mapping*/
	0,                                /*tp_hash */
	0,                                /*tp_call*/
	0,                                /*tp_str*/
	0,                                /*tp_getattro*/
	0,                                /*tp_setattro*/
	0,                                /*tp_as_buffer*/
	Py_TPFLAGS_DEFAULT,               /*tp_flags*/
	Binomial_doc,                     /*tp_doc*/
	0,                                /*tp_traverse*/
	0,                                /*tp_clear*/
	0,                                /*tp_richcompare*/
	0,                                /*tp_weaklistoffset*/
	0,                                /*tp_iter*/
	0,                                /*tp_iternext*/
	Binomial_methods,                 /*tp_methods*/
	0,                                /*tp_members*/
	0,                                /*tp_getset*/
	&UnivariateDistribution_type,     /*tp_base*/
	0,                                /*tp_dict*/
	0,                                /*tp_descr_get*/
	0,                                /*tp_descr_set*/
	0,                                /*tp_dictoffset*/
	(initproc)Binomial_init,          /*tp_init*/
	0,                                /*tp_alloc*/
	Distribution_new,                 /*tp_new*/
};

static PyGetSetDef Preconditioner_getset[] = {
	{"dim_in", (getter)Preconditioner_dim_in, 0, 0},
	{"dim_in_pre", (getter)Preconditioner_dim_in_pre, 0, 0},
	{"dim_out", (getter)Preconditioner_dim_out, 0, 0},
	{"dim_out_pre", (getter)Preconditioner_dim_out_pre, 0, 0},
	{0}
};

static PyMethodDef Preconditioner_methods[] = {
	{"inverse", (PyCFunction)Preconditioner_inverse, METH_VARARGS | METH_KEYWORDS, Preconditioner_inverse_doc},
	{"logjacobian", (PyCFunction)Preconditioner_logjacobian, METH_VARARGS | METH_KEYWORDS, Preconditioner_logjacobian_doc},
	{0}
};

PyTypeObject Preconditioner_type = {
	PyObject_HEAD_INIT(0)
	0,                                  /*ob_size*/
	"cmt.transforms.Preconditioner",    /*tp_name*/
	sizeof(PreconditionerObject),       /*tp_basicsize*/
	0,                                  /*tp_itemsize*/
	(destructor)Preconditioner_dealloc, /*tp_dealloc*/
	0,                                  /*tp_print*/
	0,                                  /*tp_getattr*/
	0,                                  /*tp_setattr*/
	0,                                  /*tp_compare*/
	0,                                  /*tp_repr*/
	0,                                  /*tp_as_number*/
	0,                                  /*tp_as_sequence*/
	0,                                  /*tp_as_mapping*/
	0,                                  /*tp_hash */
	(ternaryfunc)Preconditioner_call,   /*tp_call*/
	0,                                  /*tp_str*/
	0,                                  /*tp_getattro*/
	0,                                  /*tp_setattro*/
	0,                                  /*tp_as_buffer*/
	Py_TPFLAGS_DEFAULT,                 /*tp_flags*/
	Preconditioner_doc,                 /*tp_doc*/
	0,                                  /*tp_traverse*/
	0,                                  /*tp_clear*/
	0,                                  /*tp_richcompare*/
	0,                                  /*tp_weaklistoffset*/
	0,                                  /*tp_iter*/
	0,                                  /*tp_iternext*/
	Preconditioner_methods,             /*tp_methods*/
	0,                                  /*tp_members*/
	Preconditioner_getset,              /*tp_getset*/
	0,                                  /*tp_base*/
	0,                                  /*tp_dict*/
	0,                                  /*tp_descr_get*/
	0,                                  /*tp_descr_set*/
	0,                                  /*tp_dictoffset*/
	(initproc)Preconditioner_init,      /*tp_init*/
	0,                                  /*tp_alloc*/
	Preconditioner_new,                 /*tp_new*/
};

static PyGetSetDef AffinePreconditioner_getset[] = {
	{"mean_in", (getter)AffinePreconditioner_mean_in, 0, 0},
	{"mean_out", (getter)AffinePreconditioner_mean_out, 0, 0},
	{"pre_in", (getter)AffinePreconditioner_pre_in, 0, 0},
	{"pre_out", (getter)AffinePreconditioner_pre_out, 0, 0},
	{"predictor", (getter)AffinePreconditioner_predictor, 0, 0},
	{0}
};

static PyMethodDef AffinePreconditioner_methods[] = {
	{"__reduce__", (PyCFunction)AffinePreconditioner_reduce, METH_NOARGS, 0},
	{"__setstate__", (PyCFunction)AffinePreconditioner_setstate, METH_VARARGS, 0},
	{0}
};

PyTypeObject AffinePreconditioner_type = {
	PyObject_HEAD_INIT(0)
	0,                                     /*ob_size*/
	"cmt.transforms.AffinePreconditioner", /*tp_name*/
	sizeof(AffinePreconditionerObject),    /*tp_basicsize*/
	0,                                     /*tp_itemsize*/
	(destructor)Preconditioner_dealloc,    /*tp_dealloc*/
	0,                                     /*tp_print*/
	0,                                     /*tp_getattr*/
	0,                                     /*tp_setattr*/
	0,                                     /*tp_compare*/
	0,                                     /*tp_repr*/
	0,                                     /*tp_as_number*/
	0,                                     /*tp_as_sequence*/
	0,                                     /*tp_as_mapping*/
	0,                                     /*tp_hash */
	0,                                     /*tp_call*/
	0,                                     /*tp_str*/
	0,                                     /*tp_getattro*/
	0,                                     /*tp_setattro*/
	0,                                     /*tp_as_buffer*/
	Py_TPFLAGS_DEFAULT,                    /*tp_flags*/
	AffinePreconditioner_doc,              /*tp_doc*/
	0,                                     /*tp_traverse*/
	0,                                     /*tp_clear*/
	0,                                     /*tp_richcompare*/
	0,                                     /*tp_weaklistoffset*/
	0,                                     /*tp_iter*/
	0,                                     /*tp_iternext*/
	AffinePreconditioner_methods,          /*tp_methods*/
	0,                                     /*tp_members*/
	AffinePreconditioner_getset,           /*tp_getset*/
	&Preconditioner_type,                  /*tp_base*/
	0,                                     /*tp_dict*/
	0,                                     /*tp_descr_get*/
	0,                                     /*tp_descr_set*/
	0,                                     /*tp_dictoffset*/
	(initproc)AffinePreconditioner_init,   /*tp_init*/
	0,                                     /*tp_alloc*/
	Preconditioner_new,                    /*tp_new*/
};

static PyMethodDef AffineTransform_methods[] = {
	{"__reduce__", (PyCFunction)AffineTransform_reduce, METH_NOARGS, 0},
	{0}
};

PyTypeObject AffineTransform_type = {
	PyObject_HEAD_INIT(0)
	0,                                  /*ob_size*/
	"cmt.transforms.AffineTransform",   /*tp_name*/
	sizeof(AffineTransformObject),      /*tp_basicsize*/
	0,                                  /*tp_itemsize*/
	(destructor)Preconditioner_dealloc, /*tp_dealloc*/
	0,                                  /*tp_print*/
	0,                                  /*tp_getattr*/
	0,                                  /*tp_setattr*/
	0,                                  /*tp_compare*/
	0,                                  /*tp_repr*/
	0,                                  /*tp_as_number*/
	0,                                  /*tp_as_sequence*/
	0,                                  /*tp_as_mapping*/
	0,                                  /*tp_hash */
	0,                                  /*tp_call*/
	0,                                  /*tp_str*/
	0,                                  /*tp_getattro*/
	0,                                  /*tp_setattro*/
	0,                                  /*tp_as_buffer*/
	Py_TPFLAGS_DEFAULT,                 /*tp_flags*/
	AffineTransform_doc,                /*tp_doc*/
	0,                                  /*tp_traverse*/
	0,                                  /*tp_clear*/
	0,                                  /*tp_richcompare*/
	0,                                  /*tp_weaklistoffset*/
	0,                                  /*tp_iter*/
	0,                                  /*tp_iternext*/
	AffineTransform_methods,            /*tp_methods*/
	0,                                  /*tp_members*/
	0,                                  /*tp_getset*/
	&AffinePreconditioner_type,         /*tp_base*/
	0,                                  /*tp_dict*/
	0,                                  /*tp_descr_get*/
	0,                                  /*tp_descr_set*/
	0,                                  /*tp_dictoffset*/
	(initproc)AffineTransform_init,     /*tp_init*/
	0,                                  /*tp_alloc*/
	Preconditioner_new,                 /*tp_new*/
};

PyTypeObject WhiteningPreconditioner_type = {
	PyObject_HEAD_INIT(0)
	0,                                        /*ob_size*/
	"cmt.transforms.WhiteningPreconditioner", /*tp_name*/
	sizeof(WhiteningPreconditionerObject),    /*tp_basicsize*/
	0,                                        /*tp_itemsize*/
	(destructor)Preconditioner_dealloc,       /*tp_dealloc*/
	0,                                        /*tp_print*/
	0,                                        /*tp_getattr*/
	0,                                        /*tp_setattr*/
	0,                                        /*tp_compare*/
	0,                                        /*tp_repr*/
	0,                                        /*tp_as_number*/
	0,                                        /*tp_as_sequence*/
	0,                                        /*tp_as_mapping*/
	0,                                        /*tp_hash */
	0,                                        /*tp_call*/
	0,                                        /*tp_str*/
	0,                                        /*tp_getattro*/
	0,                                        /*tp_setattro*/
	0,                                        /*tp_as_buffer*/
	Py_TPFLAGS_DEFAULT,                       /*tp_flags*/
	WhiteningPreconditioner_doc,              /*tp_doc*/
	0,                                        /*tp_traverse*/
	0,                                        /*tp_clear*/
	0,                                        /*tp_richcompare*/
	0,                                        /*tp_weaklistoffset*/
	0,                                        /*tp_iter*/
	0,                                        /*tp_iternext*/
	0,                                        /*tp_methods*/
	0,                                        /*tp_members*/
	0,                                        /*tp_getset*/
	&AffinePreconditioner_type,               /*tp_base*/
	0,                                        /*tp_dict*/
	0,                                        /*tp_descr_get*/
	0,                                        /*tp_descr_set*/
	0,                                        /*tp_dictoffset*/
	(initproc)WhiteningPreconditioner_init,   /*tp_init*/
	0,                                        /*tp_alloc*/
	Preconditioner_new,                       /*tp_new*/
};

PyTypeObject WhiteningTransform_type = {
	PyObject_HEAD_INIT(0)
	0,                                   /*ob_size*/
	"cmt.transforms.WhiteningTransform", /*tp_name*/
	sizeof(WhiteningTransformObject),    /*tp_basicsize*/
	0,                                   /*tp_itemsize*/
	(destructor)Preconditioner_dealloc,  /*tp_dealloc*/
	0,                                   /*tp_print*/
	0,                                   /*tp_getattr*/
	0,                                   /*tp_setattr*/
	0,                                   /*tp_compare*/
	0,                                   /*tp_repr*/
	0,                                   /*tp_as_number*/
	0,                                   /*tp_as_sequence*/
	0,                                   /*tp_as_mapping*/
	0,                                   /*tp_hash */
	0,                                   /*tp_call*/
	0,                                   /*tp_str*/
	0,                                   /*tp_getattro*/
	0,                                   /*tp_setattro*/
	0,                                   /*tp_as_buffer*/
	Py_TPFLAGS_DEFAULT,                  /*tp_flags*/
	WhiteningTransform_doc,              /*tp_doc*/
	0,                                   /*tp_traverse*/
	0,                                   /*tp_clear*/
	0,                                   /*tp_richcompare*/
	0,                                   /*tp_weaklistoffset*/
	0,                                   /*tp_iter*/
	0,                                   /*tp_iternext*/
	0,                                   /*tp_methods*/
	0,                                   /*tp_members*/
	0,                                   /*tp_getset*/
	&AffineTransform_type,               /*tp_base*/
	0,                                   /*tp_dict*/
	0,                                   /*tp_descr_get*/
	0,                                   /*tp_descr_set*/
	0,                                   /*tp_dictoffset*/
	(initproc)WhiteningTransform_init,   /*tp_init*/
	0,                                   /*tp_alloc*/
	Preconditioner_new,                  /*tp_new*/
};

static PyGetSetDef PCAPreconditioner_getset[] = {
	{"eigenvalues", (getter)PCAPreconditioner_eigenvalues, 0, "Eigenvalues of the covariance of the input."},
	{0}
};

static PyMethodDef PCAPreconditioner_methods[] = {
	{"__reduce__", (PyCFunction)PCAPreconditioner_reduce, METH_NOARGS, 0},
	{0}
};

PyTypeObject PCAPreconditioner_type = {
	PyObject_HEAD_INIT(0)
	0,                                  /*ob_size*/
	"cmt.transforms.PCAPreconditioner", /*tp_name*/
	sizeof(PCAPreconditionerObject),    /*tp_basicsize*/
	0,                                  /*tp_itemsize*/
	(destructor)Preconditioner_dealloc, /*tp_dealloc*/
	0,                                  /*tp_print*/
	0,                                  /*tp_getattr*/
	0,                                  /*tp_setattr*/
	0,                                  /*tp_compare*/
	0,                                  /*tp_repr*/
	0,                                  /*tp_as_number*/
	0,                                  /*tp_as_sequence*/
	0,                                  /*tp_as_mapping*/
	0,                                  /*tp_hash */
	0,                                  /*tp_call*/
	0,                                  /*tp_str*/
	0,                                  /*tp_getattro*/
	0,                                  /*tp_setattro*/
	0,                                  /*tp_as_buffer*/
	Py_TPFLAGS_DEFAULT,                 /*tp_flags*/
	PCAPreconditioner_doc,              /*tp_doc*/
	0,                                  /*tp_traverse*/
	0,                                  /*tp_clear*/
	0,                                  /*tp_richcompare*/
	0,                                  /*tp_weaklistoffset*/
	0,                                  /*tp_iter*/
	0,                                  /*tp_iternext*/
	PCAPreconditioner_methods,          /*tp_methods*/
	0,                                  /*tp_members*/
	PCAPreconditioner_getset,           /*tp_getset*/
	&AffinePreconditioner_type,         /*tp_base*/
	0,                                  /*tp_dict*/
	0,                                  /*tp_descr_get*/
	0,                                  /*tp_descr_set*/
	0,                                  /*tp_dictoffset*/
	(initproc)PCAPreconditioner_init,   /*tp_init*/
	0,                                  /*tp_alloc*/
	Preconditioner_new,                 /*tp_new*/
};

static PyGetSetDef PCATransform_getset[] = {
	{"eigenvalues", (getter)PCATransform_eigenvalues, 0, "Eigenvalues of the covariance of the input."},
	{0}
};

static PyMethodDef PCATransform_methods[] = {
	{"__reduce__", (PyCFunction)PCATransform_reduce, METH_NOARGS, 0},
	{0}
};

PyTypeObject PCATransform_type = {
	PyObject_HEAD_INIT(0)
	0,                                  /*ob_size*/
	"cmt.transforms.PCATransform",      /*tp_name*/
	sizeof(PCATransformObject),         /*tp_basicsize*/
	0,                                  /*tp_itemsize*/
	(destructor)Preconditioner_dealloc, /*tp_dealloc*/
	0,                                  /*tp_print*/
	0,                                  /*tp_getattr*/
	0,                                  /*tp_setattr*/
	0,                                  /*tp_compare*/
	0,                                  /*tp_repr*/
	0,                                  /*tp_as_number*/
	0,                                  /*tp_as_sequence*/
	0,                                  /*tp_as_mapping*/
	0,                                  /*tp_hash */
	0,                                  /*tp_call*/
	0,                                  /*tp_str*/
	0,                                  /*tp_getattro*/
	0,                                  /*tp_setattro*/
	0,                                  /*tp_as_buffer*/
	Py_TPFLAGS_DEFAULT,                 /*tp_flags*/
	PCATransform_doc,                   /*tp_doc*/
	0,                                  /*tp_traverse*/
	0,                                  /*tp_clear*/
	0,                                  /*tp_richcompare*/
	0,                                  /*tp_weaklistoffset*/
	0,                                  /*tp_iter*/
	0,                                  /*tp_iternext*/
	PCATransform_methods,               /*tp_methods*/
	0,                                  /*tp_members*/
	PCATransform_getset,                /*tp_getset*/
	&AffineTransform_type,              /*tp_base*/
	0,                                  /*tp_dict*/
	0,                                  /*tp_descr_get*/
	0,                                  /*tp_descr_set*/
	0,                                  /*tp_dictoffset*/
	(initproc)PCATransform_init,        /*tp_init*/
	0,                                  /*tp_alloc*/
	Preconditioner_new,                 /*tp_new*/
};

static PyGetSetDef BinningTransform_getset[] = {
	{"binning", (getter)BinningTransform_binning, 0, "Binning width."},
	{0}
};

static PyMethodDef BinningTransform_methods[] = {
	{"__reduce__", (PyCFunction)BinningTransform_reduce, METH_NOARGS, 0},
	{0}
};

PyTypeObject BinningTransform_type = {
	PyObject_HEAD_INIT(0)
	0,                                  /*ob_size*/
	"cmt.transforms.BinningTransform",  /*tp_name*/
	sizeof(BinningTransformObject),     /*tp_basicsize*/
	0,                                  /*tp_itemsize*/
	(destructor)Preconditioner_dealloc, /*tp_dealloc*/
	0,                                  /*tp_print*/
	0,                                  /*tp_getattr*/
	0,                                  /*tp_setattr*/
	0,                                  /*tp_compare*/
	0,                                  /*tp_repr*/
	0,                                  /*tp_as_number*/
	0,                                  /*tp_as_sequence*/
	0,                                  /*tp_as_mapping*/
	0,                                  /*tp_hash */
	0,                                  /*tp_call*/
	0,                                  /*tp_str*/
	0,                                  /*tp_getattro*/
	0,                                  /*tp_setattro*/
	0,                                  /*tp_as_buffer*/
	Py_TPFLAGS_DEFAULT,                 /*tp_flags*/
	BinningTransform_doc,               /*tp_doc*/
	0,                                  /*tp_traverse*/
	0,                                  /*tp_clear*/
	0,                                  /*tp_richcompare*/
	0,                                  /*tp_weaklistoffset*/
	0,                                  /*tp_iter*/
	0,                                  /*tp_iternext*/
	BinningTransform_methods,           /*tp_methods*/
	0,                                  /*tp_members*/
	BinningTransform_getset,            /*tp_getset*/
	&AffineTransform_type,              /*tp_base*/
	0,                                  /*tp_dict*/
	0,                                  /*tp_descr_get*/
	0,                                  /*tp_descr_set*/
	0,                                  /*tp_dictoffset*/
	(initproc)BinningTransform_init,    /*tp_init*/
	0,                                  /*tp_alloc*/
	Preconditioner_new,                 /*tp_new*/
};

static const char* cmt_doc =
	"This module provides fast implementations of different probabilistic models.";

PyObject* seed(PyObject* self, PyObject* args, PyObject* kwds) {
	int seed;

	if(!PyArg_ParseTuple(args, "i", &seed))
		return 0;

	srand(seed);

	Py_INCREF(Py_None);
	return Py_None;
}

static PyMethodDef cmt_methods[] = {
	{"seed", (PyCFunction)seed, METH_VARARGS, 0},
	{"random_select", (PyCFunction)random_select, METH_VARARGS | METH_KEYWORDS, random_select_doc},
	{"generate_data_from_image", (PyCFunction)generate_data_from_image, METH_VARARGS | METH_KEYWORDS, generate_data_from_image_doc},
	{"generate_data_from_video", (PyCFunction)generate_data_from_video, METH_VARARGS | METH_KEYWORDS, generate_data_from_video_doc},
	{"sample_image", (PyCFunction)sample_image, METH_VARARGS | METH_KEYWORDS, sample_image_doc},
	{"sample_image_conditionally", (PyCFunction)sample_image_conditionally, METH_VARARGS | METH_KEYWORDS, sample_image_conditionally_doc},
	{"sample_labels_conditionally", (PyCFunction)sample_labels_conditionally, METH_VARARGS | METH_KEYWORDS, sample_labels_conditionally_doc},
	{"sample_video", (PyCFunction)sample_video, METH_VARARGS | METH_KEYWORDS, sample_video_doc},
	{"fill_in_image", (PyCFunction)fill_in_image, METH_VARARGS | METH_KEYWORDS, fill_in_image_doc},
	{"fill_in_image_map", (PyCFunction)fill_in_image_map, METH_VARARGS | METH_KEYWORDS, 0},
	{"extract_windows", (PyCFunction)extract_windows, METH_VARARGS | METH_KEYWORDS, extract_windows_doc},
	{"sample_spike_train", (PyCFunction)sample_spike_train, METH_VARARGS | METH_KEYWORDS, sample_spike_train_doc},
	{0}
};

PyMODINIT_FUNC init_cmt() {
	// set random seed
	timeval time;
	gettimeofday(&time, 0);
	srand(time.tv_usec * time.tv_sec);

	// initialize NumPy
	import_array();

	// create module object
	PyObject* module = Py_InitModule3("_cmt", cmt_methods, cmt_doc);

	// initialize types
	if(PyType_Ready(&AffinePreconditioner_type) < 0)
		return;
	if(PyType_Ready(&AffineTransform_type) < 0)
		return;
	if(PyType_Ready(&Bernoulli_type) < 0)
		return;
	if(PyType_Ready(&BinningTransform_type) < 0)
		return;
	if(PyType_Ready(&Binomial_type) < 0)
		return;
	if(PyType_Ready(&BlobNonlinearity_type) < 0)
		return;
	if(PyType_Ready(&CD_type) < 0)
		return;
	if(PyType_Ready(&DifferentiableNonlinearity_type) < 0)
		return;
	if(PyType_Ready(&Distribution_type) < 0)
		return;
	if(PyType_Ready(&ExponentialFunction_type) < 0)
		return;
	if(PyType_Ready(&FVBN_type) < 0)
		return;
	if(PyType_Ready(&GLM_type) < 0)
		return;
	if(PyType_Ready(&GSM_type) < 0)
		return;
	if(PyType_Ready(&InvertibleNonlinearity_type) < 0)
		return;
	if(PyType_Ready(&HistogramNonlinearity_type) < 0)
		return;
	if(PyType_Ready(&LogisticFunction_type) < 0)
		return;
	if(PyType_Ready(&MCBM_type) < 0)
		return;
	if(PyType_Ready(&MCGSM_type) < 0)
		return;
	if(PyType_Ready(&Mixture_type) < 0)
		return;
	if(PyType_Ready(&MixtureComponent_type) < 0)
		return;
	if(PyType_Ready(&MLR_type) < 0)
		return;
	if(PyType_Ready(&MoGSM_type) < 0)
		return;
	if(PyType_Ready(&Nonlinearity_type) < 0)
		return;
	if(PyType_Ready(&PatchMCBM_type) < 0)
		return;
	if(PyType_Ready(&PatchMCGSM_type) < 0)
		return;
	if(PyType_Ready(&PatchModel_type) < 0)
		return;
	if(PyType_Ready(&PCAPreconditioner_type) < 0)
		return;
	if(PyType_Ready(&PCATransform_type) < 0)
		return;
	if(PyType_Ready(&Poisson_type) < 0)
		return;
	if(PyType_Ready(&Preconditioner_type) < 0)
		return;
	if(PyType_Ready(&STM_type) < 0)
		return;
	if(PyType_Ready(&TanhBlobNonlinearity_type) < 0)
		return;
	if(PyType_Ready(&TrainableNonlinearity_type) < 0)
		return;
	if(PyType_Ready(&UnivariateDistribution_type) < 0)
		return;
	if(PyType_Ready(&WhiteningPreconditioner_type) < 0)
		return;
	if(PyType_Ready(&WhiteningTransform_type) < 0)
		return;

	// initialize Eigen
	Eigen::initParallel();

	// add types to module
	Py_INCREF(&AffinePreconditioner_type);
	Py_INCREF(&AffineTransform_type);
	Py_INCREF(&Bernoulli_type);
	Py_INCREF(&BinningTransform_type);
	Py_INCREF(&Binomial_type);
	Py_INCREF(&BlobNonlinearity_type);
	Py_INCREF(&CD_type);
	Py_INCREF(&DifferentiableNonlinearity_type);
	Py_INCREF(&Distribution_type);
	Py_INCREF(&ExponentialFunction_type);
	Py_INCREF(&FVBN_type);
	Py_INCREF(&GLM_type);
	Py_INCREF(&GSM_type);
	Py_INCREF(&HistogramNonlinearity_type);
	Py_INCREF(&InvertibleNonlinearity_type);
	Py_INCREF(&LogisticFunction_type);
	Py_INCREF(&MCBM_type);
	Py_INCREF(&MCGSM_type);
	Py_INCREF(&Mixture_type);
	Py_INCREF(&MixtureComponent_type);
	Py_INCREF(&MLR_type);
	Py_INCREF(&MoGSM_type);
	Py_INCREF(&Nonlinearity_type);
	Py_INCREF(&PCAPreconditioner_type);
	Py_INCREF(&PCATransform_type);
	Py_INCREF(&PatchMCBM_type);
	Py_INCREF(&PatchMCGSM_type);
	Py_INCREF(&PatchModel_type);
	Py_INCREF(&Poisson_type);
	Py_INCREF(&Preconditioner_type);
	Py_INCREF(&STM_type);
	Py_INCREF(&TanhBlobNonlinearity_type);
	Py_INCREF(&TrainableNonlinearity_type);
	Py_INCREF(&UnivariateDistribution_type);
	Py_INCREF(&WhiteningPreconditioner_type);
	Py_INCREF(&WhiteningTransform_type);

	PyModule_AddObject(module, "AffinePreconditioner", reinterpret_cast<PyObject*>(&AffinePreconditioner_type));
	PyModule_AddObject(module, "AffineTransform", reinterpret_cast<PyObject*>(&AffineTransform_type));
	PyModule_AddObject(module, "Bernoulli", reinterpret_cast<PyObject*>(&Bernoulli_type));
	PyModule_AddObject(module, "BinningTransform", reinterpret_cast<PyObject*>(&BinningTransform_type));
	PyModule_AddObject(module, "Binomial", reinterpret_cast<PyObject*>(&Binomial_type));
	PyModule_AddObject(module, "BlobNonlinearity", reinterpret_cast<PyObject*>(&BlobNonlinearity_type));
	PyModule_AddObject(module, "ConditionalDistribution", reinterpret_cast<PyObject*>(&CD_type));
	PyModule_AddObject(module, "DifferentiableNonlinearity", reinterpret_cast<PyObject*>(&DifferentiableNonlinearity_type));
	PyModule_AddObject(module, "Distribution", reinterpret_cast<PyObject*>(&Distribution_type));
	PyModule_AddObject(module, "ExponentialFunction", reinterpret_cast<PyObject*>(&ExponentialFunction_type));
	PyModule_AddObject(module, "FVBN", reinterpret_cast<PyObject*>(&FVBN_type));
	PyModule_AddObject(module, "GLM", reinterpret_cast<PyObject*>(&GLM_type));
	PyModule_AddObject(module, "GSM", reinterpret_cast<PyObject*>(&GSM_type));
	PyModule_AddObject(module, "HistogramNonlinearity", reinterpret_cast<PyObject*>(&HistogramNonlinearity_type));
	PyModule_AddObject(module, "InvertibleNonlinearity", reinterpret_cast<PyObject*>(&InvertibleNonlinearity_type));
	PyModule_AddObject(module, "LogisticFunction", reinterpret_cast<PyObject*>(&LogisticFunction_type));
	PyModule_AddObject(module, "MCBM", reinterpret_cast<PyObject*>(&MCBM_type));
	PyModule_AddObject(module, "MCGSM", reinterpret_cast<PyObject*>(&MCGSM_type));
	PyModule_AddObject(module, "Mixture", reinterpret_cast<PyObject*>(&Mixture_type));
	PyModule_AddObject(module, "MixtureComponent", reinterpret_cast<PyObject*>(&MixtureComponent_type));
	PyModule_AddObject(module, "MLR", reinterpret_cast<PyObject*>(&MLR_type));
	PyModule_AddObject(module, "MoGSM", reinterpret_cast<PyObject*>(&MoGSM_type));
	PyModule_AddObject(module, "Nonlinearity", reinterpret_cast<PyObject*>(&Nonlinearity_type));
	PyModule_AddObject(module, "PCAPreconditioner", reinterpret_cast<PyObject*>(&PCAPreconditioner_type));
	PyModule_AddObject(module, "PCATransform", reinterpret_cast<PyObject*>(&PCATransform_type));
	PyModule_AddObject(module, "PatchMCBM", reinterpret_cast<PyObject*>(&PatchMCBM_type));
	PyModule_AddObject(module, "PatchMCGSM", reinterpret_cast<PyObject*>(&PatchMCGSM_type));
	PyModule_AddObject(module, "PatchModel", reinterpret_cast<PyObject*>(&PatchModel_type));
	PyModule_AddObject(module, "Poisson", reinterpret_cast<PyObject*>(&Poisson_type));
	PyModule_AddObject(module, "Preconditioner", reinterpret_cast<PyObject*>(&Preconditioner_type));
	PyModule_AddObject(module, "STM", reinterpret_cast<PyObject*>(&STM_type));
	PyModule_AddObject(module, "TanhBlobNonlinearity", reinterpret_cast<PyObject*>(&TanhBlobNonlinearity_type));
	PyModule_AddObject(module, "TrainableNonlinearity", reinterpret_cast<PyObject*>(&TrainableNonlinearity_type));
	PyModule_AddObject(module, "UnivariateDistribution", reinterpret_cast<PyObject*>(&UnivariateDistribution_type));
	PyModule_AddObject(module, "WhiteningPreconditioner", reinterpret_cast<PyObject*>(&WhiteningPreconditioner_type));
	PyModule_AddObject(module, "WhiteningTransform", reinterpret_cast<PyObject*>(&WhiteningTransform_type));
}
