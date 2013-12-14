#include "toolsinterface.h"
#include "mcgsminterface.h"
#include "preconditionerinterface.h"
#include "conditionaldistributioninterface.h"

#include "cmt/utils"
using CMT::Exception;
using CMT::randomSelect;

#include "cmt/models"
using CMT::ConditionalDistribution;

#include "cmt/tools"
using CMT::generateDataFromImage;
using CMT::generateDataFromVideo;
using CMT::sampleImage;
using CMT::sampleVideo;
using CMT::fillInImage;
using CMT::fillInImageMAP;
using CMT::extractWindows;
using CMT::sampleSpikeTrain;

#include <utility>
using std::pair;
using std::make_pair;

#include <set>
using std::set;

#include <new>
using std::bad_alloc;

const char* random_select_doc =
	"random_select(k, n)\n"
	"\n"
	"Randomly selects $k$ out of $n$ elements.\n"
	"\n"
	"@type  k: C{int}\n"
	"@param k: the number of elements to pick\n"
	"\n"
	"@type  n: C{int}\n"
	"@param n: the number of elements to pick from\n"
	"\n"
	"@rtype: C{list}\n"
	"@return: a list of $k$ indices";

PyObject* random_select(PyObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"k", "n", 0};

	int n;
	int k;

	if(!PyArg_ParseTupleAndKeywords(args, kwds, "ii", const_cast<char**>(kwlist), &k, &n))
		return 0;

	try {
		set<int> indices = randomSelect(k, n);

		PyObject* list = PyList_New(indices.size());
		
		int i = 0;

		for(set<int>::iterator iter = indices.begin(); iter != indices.end(); ++iter, ++i)
			PyList_SetItem(list, i, PyInt_FromLong(*iter));

		return list;
	} catch(Exception& exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}
	
	return 0;
}



const char* generate_data_from_image_doc =
	"generate_data_from_image(img, input_mask, output_mask, num_samples=0)\n"
	"\n"
	"Uniformly samples inputs and outputs for conditional models from images.\n"
	"\n"
	"If no number of samples is specified, all possible inputs and outputs are\n"
	"extracted from the image and returned in row-major order.\n"
	"\n"
	"@type  img: C{ndarray}\n"
	"@param img: an array representing a grayscale or color image\n"
	"\n"
	"@type  input_mask: C{ndarray}\n"
	"@param input_mask: a Boolean array describing the input pixels\n"
	"\n"
	"@type  output_mask: C{ndarray}\n"
	"@param output_mask: a Boolean array describing the output pixels\n"
	"\n"
	"@type  num_samples: C{int}\n"
	"@param num_samples: the number of generated input/output pairs\n"
	"\n"
	"@rtype: C{tuple}\n"
	"@return: the input and output vectors stored in columns";

PyObject* generate_data_from_image(PyObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"img", "input_mask", "output_mask", "num_samples", 0};

	PyObject* img;
	PyObject* input_mask;
	PyObject* output_mask;
	int num_samples = 0;

	if(!PyArg_ParseTupleAndKeywords(args, kwds, "OOO|i", const_cast<char**>(kwlist),
		&img, &input_mask, &output_mask, &num_samples))
		return 0;

	// make sure data is stored in NumPy array
	img = PyArray_FROM_OTF(img, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
	input_mask = PyArray_FROM_OTF(input_mask, NPY_BOOL, NPY_F_CONTIGUOUS | NPY_ALIGNED);
	output_mask = PyArray_FROM_OTF(output_mask, NPY_BOOL, NPY_F_CONTIGUOUS | NPY_ALIGNED);

	if(!img) {
		PyErr_SetString(PyExc_TypeError, "The image has to be given as an array.");
		return 0;
	}

	if(!input_mask || !output_mask) {
		Py_DECREF(img);
		Py_XDECREF(input_mask);
		Py_XDECREF(output_mask);
		PyErr_SetString(PyExc_TypeError, "Masks have to be given as Boolean arrays.");
		return 0;
	}

	try {
		pair<ArrayXXd, ArrayXXd> dataPair;
		
		if(PyArray_NDIM(img) > 2)
			if(PyArray_NDIM(input_mask) > 2 && PyArray_NDIM(output_mask) > 2)
				// multi-channel image and multi-channel masks
				dataPair = generateDataFromImage(
					PyArray_ToArraysXXd(img),
					PyArray_ToArraysXXb(input_mask),
					PyArray_ToArraysXXb(output_mask),
					num_samples);
			else
				// multi-channel image and single-channel masks
				dataPair = generateDataFromImage(
					PyArray_ToArraysXXd(img),
					PyArray_ToMatrixXb(input_mask),
					PyArray_ToMatrixXb(output_mask),
					num_samples);
		else
			// single-channel image and single-channel masks
			dataPair = generateDataFromImage(
				PyArray_ToMatrixXd(img),
				PyArray_ToMatrixXb(input_mask),
				PyArray_ToMatrixXb(output_mask),
				num_samples);

		PyObject* xvalues = PyArray_FromMatrixXd(dataPair.first);
		PyObject* yvalues = PyArray_FromMatrixXd(dataPair.second);

		PyObject* data = Py_BuildValue("(OO)",
			xvalues,
			yvalues);

		Py_DECREF(img);
		Py_DECREF(xvalues);
		Py_DECREF(yvalues);
		Py_DECREF(input_mask);
		Py_DECREF(output_mask);

		return data;

	} catch(Exception& exception) {
		Py_DECREF(img);
		Py_DECREF(input_mask);
		Py_DECREF(output_mask);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	} catch(bad_alloc&) {
		Py_DECREF(img);
		Py_DECREF(input_mask);
		Py_DECREF(output_mask);
		PyErr_SetString(PyExc_RuntimeError, "Could not allocate memory.");
		return 0;
	}

	Py_DECREF(img);
	Py_DECREF(input_mask);
	Py_DECREF(output_mask);

	return 0;
}



const char* generate_data_from_video_doc =
	"generate_data_from_video(video, input_mask, output_mask, num_samples=0)\n"
	"\n"
	"Uniformly samples inputs and outputs for conditional models from videos.\n"
	"\n"
	"If no number of samples is specified, all possible inputs and outputs are\n"
	"extracted from the image and returned in row-major order.\n"
	"\n"
	"@type  video: C{ndarray}\n"
	"@param video: a three-dimensional array representing the video\n"
	"\n"
	"@type  input_mask: C{ndarray}\n"
	"@param input_mask: a three-dimensioal Boolean array describing the input pixels\n"
	"\n"
	"@type  output_mask: C{ndarray}\n"
	"@param output_mask: a three-dimensioal Boolean array describing the output pixels\n"
	"\n"
	"@type  num_samples: C{int}\n"
	"@param num_samples: the number of generated input/output pairs\n"
	"\n"
	"@rtype: C{tuple}\n"
	"@return: the input and output vectors stored in columns";

PyObject* generate_data_from_video(PyObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"video", "input_mask", "output_mask", "num_samples", 0};

	PyObject* video;
	PyObject* input_mask;
	PyObject* output_mask;
	int num_samples = 0;

	if(!PyArg_ParseTupleAndKeywords(args, kwds, "OOO|i", const_cast<char**>(kwlist),
		&video, &input_mask, &output_mask, &num_samples))
		return 0;

	// make sure data is stored in NumPy array
	video = PyArray_FROM_OTF(video, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
	input_mask = PyArray_FROM_OTF(input_mask, NPY_BOOL, NPY_F_CONTIGUOUS | NPY_ALIGNED);
	output_mask = PyArray_FROM_OTF(output_mask, NPY_BOOL, NPY_F_CONTIGUOUS | NPY_ALIGNED);

	if(!video) {
		PyErr_SetString(PyExc_TypeError, "The initial video has to be given as an array.");
		return 0;
	}

	if(!input_mask || !output_mask) {
		Py_DECREF(video);
		Py_XDECREF(input_mask);
		Py_XDECREF(output_mask);
		PyErr_SetString(PyExc_TypeError, "Masks have to be given as Boolean arrays.");
		return 0;
	}

	try {
		pair<ArrayXXd, ArrayXXd> dataPair;
		
		dataPair = generateDataFromVideo(
			PyArray_ToArraysXXd(video),
			PyArray_ToArraysXXb(input_mask),
			PyArray_ToArraysXXb(output_mask),
			num_samples);

		PyObject* xvalues = PyArray_FromMatrixXd(dataPair.first);
		PyObject* yvalues = PyArray_FromMatrixXd(dataPair.second);

		PyObject* data = Py_BuildValue("(OO)",
			xvalues,
			yvalues);

		Py_DECREF(video);
		Py_DECREF(xvalues);
		Py_DECREF(yvalues);
		Py_DECREF(input_mask);
		Py_DECREF(output_mask);

		return data;

	} catch(Exception& exception) {
		Py_DECREF(video);
		Py_DECREF(input_mask);
		Py_DECREF(output_mask);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	} catch(bad_alloc&) {
		Py_DECREF(video);
		Py_DECREF(input_mask);
		Py_DECREF(output_mask);
		PyErr_SetString(PyExc_RuntimeError, "Could not allocate memory.");
		return 0;
	}

	Py_DECREF(video);
	Py_DECREF(input_mask);
	Py_DECREF(output_mask);
	return 0;
}


const char* sample_image_doc =
	"sample_image(img, model, input_mask, output_mask, preconditioner=None)\n"
	"\n"
	"Generates an image using a conditional distribution. The initial image passed to\n"
	"this function is used to initialize the boundaries and is also used to determine\n"
	"the size and length of the image to be generated.\n"
	"\n"
	"@type  img: C{ndarray}\n"
	"@param img: initialization of image\n"
	"\n"
	"@type  model: L{ConditionalDistribution<models.ConditionalDistribution>}\n"
	"@param model: a conditional distribution such as an L{MCGSM<models.MCGSM>}\n"
	"\n"
	"@type  input_mask: C{ndarray}\n"
	"@param input_mask: a Boolean array describing the input pixels\n"
	"\n"
	"@type  output_mask: C{ndarray}\n"
	"@param output_mask: a Boolean array describing the output pixels\n"
	"\n"
	"@type  preconditioner: L{Preconditioner<transforms.Preconditioner>}\n"
	"@param preconditioner: transforms the input before feeding it into the model\n"
	"\n"
	"@rtype: C{ndarray}\n"
	"@return: the sampled image";

PyObject* sample_image(PyObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"img", "model", "input_mask", "output_mask", "preconditioner", 0};

	PyObject* img;
	PyObject* modelObj;
	PyObject* input_mask;
	PyObject* output_mask;
	PyObject* preconditionerObj = 0;

	if(!PyArg_ParseTupleAndKeywords(args, kwds, "OO!OO|O!", const_cast<char**>(kwlist),
		&img, &CD_type, &modelObj, &input_mask, &output_mask, &Preconditioner_type, &preconditionerObj))
		return 0;

	const ConditionalDistribution& model = *reinterpret_cast<CDObject*>(modelObj)->cd;

	Preconditioner* preconditioner = preconditionerObj ?
		reinterpret_cast<PreconditionerObject*>(preconditionerObj)->preconditioner : 0;

	// make sure data is stored in NumPy array
	img = PyArray_FROM_OTF(img, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
	input_mask = PyArray_FROM_OTF(input_mask, NPY_BOOL, NPY_F_CONTIGUOUS | NPY_ALIGNED);
	output_mask = PyArray_FROM_OTF(output_mask, NPY_BOOL, NPY_F_CONTIGUOUS | NPY_ALIGNED);

	if(!img) {
		Py_XDECREF(input_mask);
		Py_XDECREF(output_mask);
		PyErr_SetString(PyExc_TypeError, "The initial image has to be given as an array.");
		return 0;
	}

	if(!input_mask || !output_mask) {
		Py_DECREF(img);
		Py_XDECREF(input_mask);
		Py_XDECREF(output_mask);
		PyErr_SetString(PyExc_TypeError, "Masks have to be given as Boolean arrays.");
		return 0;
	}

	try {
		PyObject* imgSample;
		if(PyArray_NDIM(img) > 2) {
			if(PyArray_NDIM(input_mask) > 2 && PyArray_NDIM(output_mask) > 2) {
				// multi-channel image and multi-channel masks
				imgSample = PyArray_FromArraysXXd(
					sampleImage(
						PyArray_ToArraysXXd(img),
						model,
						PyArray_ToArraysXXb(input_mask),
						PyArray_ToArraysXXb(output_mask),
						preconditioner));
			} else {
				// multi-channel image and single-channel masks
				imgSample = PyArray_FromArraysXXd(
					sampleImage(
						PyArray_ToArraysXXd(img),
						model,
						PyArray_ToMatrixXb(input_mask),
						PyArray_ToMatrixXb(output_mask),
						preconditioner));
			}
		} else {
			if(PyArray_NDIM(input_mask) > 2 || PyArray_NDIM(output_mask) > 2)
				throw Exception("You cannot use multi-channel masks with single-channel images.");

			// single-channel image and single-channel masks
			imgSample = PyArray_FromMatrixXd(
				sampleImage(
					PyArray_ToMatrixXd(img),
					model,
					PyArray_ToMatrixXb(input_mask),
					PyArray_ToMatrixXb(output_mask),
					preconditioner));
		}

		Py_DECREF(img);
		Py_DECREF(input_mask);
		Py_DECREF(output_mask);

		return imgSample;

	} catch(Exception& exception) {
		Py_DECREF(img);
		Py_DECREF(input_mask);
		Py_DECREF(output_mask);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	} catch(bad_alloc&) {
		Py_DECREF(img);
		Py_DECREF(input_mask);
		Py_DECREF(output_mask);
		PyErr_SetString(PyExc_RuntimeError, "Could not allocate memory.");
		return 0;
	}

	Py_DECREF(img);
	Py_DECREF(input_mask);
	Py_DECREF(output_mask);
	return 0;
}



const char* sample_image_conditionally_doc =
	"sample_image(img, model, labels, input_mask, output_mask, preconditioner=None, num_iter=10, initialize=False)\n"
	"\n"
	"Conditionally samples an image from an L{MCGSM<models.MCGSM>} using Metropolis-within-Gibbs\n"
	"sampling. The image passed to this function is used as the initialization of the Gibbs sampler.\n"
	"\n"
	"@type  img: C{ndarray}\n"
	"@param img: initialization of image\n"
	"\n"
	"@type  labels: C{ndarray}\n"
	"@param labels: labels as generated by L{sample_labels_conditionally}\n"
	"\n"
	"@type  model: L{MCGSM<models.MCGSM>}\n"
	"@param model: the model defining the joint distribution over labels and images\n"
	"\n"
	"@type  input_mask: C{ndarray}\n"
	"@param input_mask: a Boolean array describing the input pixels\n"
	"\n"
	"@type  output_mask: C{ndarray}\n"
	"@param output_mask: a Boolean array describing the output pixels\n"
	"\n"
	"@type  preconditioner: L{Preconditioner<transforms.Preconditioner>}\n"
	"@param preconditioner: transforms the input before feeding it into the model\n"
	"\n"
	"@type  num_iter: C{int}\n"
	"@param num_iter: the number of Gibbs updates of each pixel\n"
	"\n"
	"@type  initialize: C{bool}\n"
	"@param initialize: if true, accept all proposed pixel updates in the first iteration\n"
	"\n"
	"@rtype: C{ndarray}\n"
	"@return: the sampled image";

PyObject* sample_image_conditionally(PyObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {
		"img",
		"labels",
		"model",
		"input_mask",
		"output_mask",
		"preconditioner",
		"num_iter",
		"initialize", 0};

	PyObject* img;
	PyObject* labels;
	PyObject* modelObj;
	PyObject* input_mask;
	PyObject* output_mask;
	PyObject* preconditionerObj = 0;
	int num_iter = 10;
	bool initialize = false;

	if(!PyArg_ParseTupleAndKeywords(args, kwds, "OOO!OO|O!ib", const_cast<char**>(kwlist),
		&img, 
		&labels,
		&MCGSM_type, &modelObj,
		&input_mask,
		&output_mask,
		&Preconditioner_type, &preconditionerObj,
		&num_iter,
		&initialize))
		return 0;

	const MCGSM& model = *reinterpret_cast<MCGSMObject*>(modelObj)->mcgsm;

	Preconditioner* preconditioner = preconditionerObj ?
		reinterpret_cast<PreconditionerObject*>(preconditionerObj)->preconditioner : 0;

	// make sure data is stored in NumPy array
	img = PyArray_FROM_OTF(img, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
	labels = PyArray_FROM_OTF(labels, NPY_INT64, NPY_F_CONTIGUOUS | NPY_ALIGNED);
	input_mask = PyArray_FROM_OTF(input_mask, NPY_BOOL, NPY_F_CONTIGUOUS | NPY_ALIGNED);
	output_mask = PyArray_FROM_OTF(output_mask, NPY_BOOL, NPY_F_CONTIGUOUS | NPY_ALIGNED);

	if(!img) {
		Py_XDECREF(labels);
		Py_XDECREF(input_mask);
		Py_XDECREF(output_mask);
		PyErr_SetString(PyExc_TypeError, "The initial image has to be given as an array.");
		return 0;
	}

	if(!labels) {
		Py_DECREF(img);
		Py_XDECREF(input_mask);
		Py_XDECREF(output_mask);
		PyErr_SetString(PyExc_TypeError, "The labels have to be stored in an integer array.");
		return 0;
	}

	if(!input_mask || !output_mask) {
		Py_DECREF(img);
		Py_DECREF(labels);
		Py_XDECREF(input_mask);
		Py_XDECREF(output_mask);
		PyErr_SetString(PyExc_TypeError, "Masks have to be given as Boolean arrays.");
		return 0;
	}

	try {
		PyObject* imgSample;

//		if(PyArray_NDIM(img) > 2) {
//			if(PyArray_NDIM(input_mask) > 2 && PyArray_NDIM(output_mask) > 2) {
//				// multi-channel image and multi-channel masks
//				imgSample = PyArray_FromArraysXXd(
//					sampleImage(
//						PyArray_ToArraysXXd(img),
//						model,
//						PyArray_ToArraysXXb(input_mask),
//						PyArray_ToArraysXXb(output_mask),
//						preconditioner));
//			} else {
//				// multi-channel image and single-channel masks
//				imgSample = PyArray_FromArraysXXd(
//					sampleImage(
//						PyArray_ToArraysXXd(img),
//						model,
//						PyArray_ToMatrixXb(input_mask),
//						PyArray_ToMatrixXb(output_mask),
//						preconditioner));
//			}
//		} else {
//			if(PyArray_NDIM(input_mask) > 2 || PyArray_NDIM(output_mask) > 2)
//				throw Exception("You cannot use multi-channel masks with single-channel images.");
//
			// single-channel image and single-channel masks
			imgSample = PyArray_FromMatrixXd(
				sampleImageConditionally(
					PyArray_ToMatrixXd(img),
					PyArray_ToMatrixXi(labels),
					model,
					PyArray_ToMatrixXb(input_mask),
					PyArray_ToMatrixXb(output_mask),
					preconditioner,
					num_iter,
					initialize));
//		}

		Py_DECREF(img);
		Py_DECREF(labels);
		Py_DECREF(input_mask);
		Py_DECREF(output_mask);

		return imgSample;

	} catch(Exception& exception) {
		Py_DECREF(img);
		Py_DECREF(labels);
		Py_DECREF(input_mask);
		Py_DECREF(output_mask);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	} catch(bad_alloc&) {
		Py_DECREF(img);
		Py_DECREF(labels);
		Py_DECREF(input_mask);
		Py_DECREF(output_mask);
		PyErr_SetString(PyExc_RuntimeError, "Could not allocate memory.");
		return 0;
	}

	Py_DECREF(img);
	Py_DECREF(labels);
	Py_DECREF(input_mask);
	Py_DECREF(output_mask);
	return 0;
}



const char* sample_labels_conditionally_doc =
	"sample_labels_conditionally(img, model, input_mask, output_mask, preconditioner=None)\n"
	"\n"
	"Samples component labels from an L{MCGSM<models.MCGSM>} for a given image.\n"
	"\n"
	"@type  img: C{ndarray}\n"
	"@param img: initialization of image\n"
	"\n"
	"@type  model: L{MCGSM<models.MCGSM>}\n"
	"@param model: the model defining the joint distribution over labels and images\n"
	"\n"
	"@type  input_mask: C{ndarray}\n"
	"@param input_mask: a Boolean array describing the input pixels\n"
	"\n"
	"@type  output_mask: C{ndarray}\n"
	"@param output_mask: a Boolean array describing the output pixels\n"
	"\n"
	"@type  preconditioner: L{Preconditioner<transforms.Preconditioner>}\n"
	"@param preconditioner: transforms the input before feeding it into the model\n"
	"\n"
	"@rtype: C{ndarray}\n"
	"@return: sampled component labels";

PyObject* sample_labels_conditionally(PyObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {
		"img",
		"model",
		"input_mask",
		"output_mask",
		"preconditioner", 0};

	PyObject* img;
	PyObject* modelObj;
	PyObject* input_mask;
	PyObject* output_mask;
	PyObject* preconditionerObj = 0;

	if(!PyArg_ParseTupleAndKeywords(args, kwds, "OO!OO|O!", const_cast<char**>(kwlist),
		&img, 
		&MCGSM_type, &modelObj,
		&input_mask,
		&output_mask,
		&Preconditioner_type, &preconditionerObj))
		return 0;

	const MCGSM& model = *reinterpret_cast<MCGSMObject*>(modelObj)->mcgsm;

	Preconditioner* preconditioner = preconditionerObj ?
		reinterpret_cast<PreconditionerObject*>(preconditionerObj)->preconditioner : 0;

	// make sure data is stored in NumPy array
	img = PyArray_FROM_OTF(img, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
	input_mask = PyArray_FROM_OTF(input_mask, NPY_BOOL, NPY_F_CONTIGUOUS | NPY_ALIGNED);
	output_mask = PyArray_FROM_OTF(output_mask, NPY_BOOL, NPY_F_CONTIGUOUS | NPY_ALIGNED);

	if(!img) {
		Py_XDECREF(input_mask);
		Py_XDECREF(output_mask);
		PyErr_SetString(PyExc_TypeError, "The initial image has to be given as an array.");
		return 0;
	}

	if(!input_mask || !output_mask) {
		Py_DECREF(img);
		Py_XDECREF(input_mask);
		Py_XDECREF(output_mask);
		PyErr_SetString(PyExc_TypeError, "Masks have to be given as Boolean arrays.");
		return 0;
	}

	try {
		PyObject* labels;

//		if(PyArray_NDIM(img) > 2) {
//			if(PyArray_NDIM(input_mask) > 2 && PyArray_NDIM(output_mask) > 2) {
//				// multi-channel image and multi-channel masks
//				imgSample = PyArray_FromArraysXXd(
//					sampleImage(
//						PyArray_ToArraysXXd(img),
//						model,
//						PyArray_ToArraysXXb(input_mask),
//						PyArray_ToArraysXXb(output_mask),
//						preconditioner));
//			} else {
//				// multi-channel image and single-channel masks
//				imgSample = PyArray_FromArraysXXd(
//					sampleImage(
//						PyArray_ToArraysXXd(img),
//						model,
//						PyArray_ToMatrixXb(input_mask),
//						PyArray_ToMatrixXb(output_mask),
//						preconditioner));
//			}
//		} else {
//			if(PyArray_NDIM(input_mask) > 2 || PyArray_NDIM(output_mask) > 2)
//				throw Exception("You cannot use multi-channel masks with single-channel images.");
//
			// single-channel image and single-channel masks
			labels = PyArray_FromMatrixXi(
				sampleLabelsConditionally(
					PyArray_ToMatrixXd(img),
					model,
					PyArray_ToMatrixXb(input_mask),
					PyArray_ToMatrixXb(output_mask),
					preconditioner));
//		}

		Py_DECREF(img);
		Py_DECREF(input_mask);
		Py_DECREF(output_mask);

		return labels;

	} catch(Exception& exception) {
		Py_DECREF(img);
		Py_DECREF(input_mask);
		Py_DECREF(output_mask);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	} catch(bad_alloc&) {
		Py_DECREF(img);
		Py_DECREF(input_mask);
		Py_DECREF(output_mask);
		PyErr_SetString(PyExc_RuntimeError, "Could not allocate memory.");
		return 0;
	}

	Py_DECREF(img);
	Py_DECREF(input_mask);
	Py_DECREF(output_mask);
	return 0;
}



const char* sample_video_doc =
	"sample_video(video, model, input_mask, output_mask, preconditioner=None)\n"
	"\n"
	"Generates a video using a conditional distribution. The initial video passed to\n"
	"this function is used to initialize the boundaries and is also used to determine\n"
	"the size and length of the video to be generated.\n"
	"\n"
	"@type  video: C{ndarray}\n"
	"@param video: initialization of video\n"
	"\n"
	"@type  model: L{ConditionalDistribution<models.ConditionalDistribution>}\n"
	"@param model: a conditional distribution such as an L{MCGSM<models.MCGSM>}\n"
	"\n"
	"@type  input_mask: C{ndarray}\n"
	"@param input_mask: a three-dimensional Boolean array describing the input pixels\n"
	"\n"
	"@type  output_mask: C{ndarray}\n"
	"@param output_mask: a three-dimensional Boolean array describing the output pixels\n"
	"\n"
	"@type  preconditioner: L{Preconditioner<transforms.Preconditioner>}\n"
	"@param preconditioner: transforms the input before feeding it into the model\n"
	"\n"
	"@rtype: C{ndarray}\n"
	"@return: the sampled video";

PyObject* sample_video(PyObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"video", "model", "input_mask", "output_mask", "preconditioner", 0};

	PyObject* video;
	PyObject* modelObj;
	PyObject* input_mask;
	PyObject* output_mask;
	PyObject* preconditionerObj = 0;

	if(!PyArg_ParseTupleAndKeywords(args, kwds, "OO!OO|O!", const_cast<char**>(kwlist),
		&video, &CD_type, &modelObj, &input_mask, &output_mask, &Preconditioner_type, &preconditionerObj))
		return 0;

	const ConditionalDistribution& model = *reinterpret_cast<CDObject*>(modelObj)->cd;

	Preconditioner* preconditioner = preconditionerObj ?
		reinterpret_cast<PreconditionerObject*>(preconditionerObj)->preconditioner : 0;

	// make sure data is stored in NumPy array
	video = PyArray_FROM_OTF(video, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
	input_mask = PyArray_FROM_OTF(input_mask, NPY_BOOL, NPY_F_CONTIGUOUS | NPY_ALIGNED);
	output_mask = PyArray_FROM_OTF(output_mask, NPY_BOOL, NPY_F_CONTIGUOUS | NPY_ALIGNED);

	if(!video) {
		Py_XDECREF(input_mask);
		Py_XDECREF(output_mask);
		PyErr_SetString(PyExc_TypeError, "The initial video has to be given as an array.");
		return 0;
	}

	if(!input_mask || !output_mask) {
		Py_DECREF(video);
		Py_XDECREF(input_mask);
		Py_XDECREF(output_mask);
		PyErr_SetString(PyExc_TypeError, "Masks have to be given as Boolean arrays.");
		return 0;
	}

	try {
		PyObject* videoSample;

		videoSample = PyArray_FromArraysXXd(
			sampleVideo(
				PyArray_ToArraysXXd(video),
				model,
				PyArray_ToArraysXXb(input_mask),
				PyArray_ToArraysXXb(output_mask),
				preconditioner));

		Py_DECREF(video);
		Py_DECREF(input_mask);
		Py_DECREF(output_mask);

		return videoSample;

	} catch(Exception& exception) {
		Py_DECREF(video);
		Py_DECREF(input_mask);
		Py_DECREF(output_mask);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	} catch(bad_alloc&) {
		Py_DECREF(video);
		Py_DECREF(input_mask);
		Py_DECREF(output_mask);
		PyErr_SetString(PyExc_RuntimeError, "Could not allocate memory.");
		return 0;
	}

	Py_DECREF(video);
	Py_DECREF(input_mask);
	Py_DECREF(output_mask);

	return 0;
}



const char* fill_in_image_doc =
	"fill_in_image(img, model, input_mask, output_mask, fmask, preconditioner=None, num_iter=10, num_steps=100)\n"
	"\n"
	"Samples pixels of an image conditioned on all other pixels.\n"
	"\n"
	"@type  img: C{ndarray}\n"
	"@param img: the image with the missing pixels initialized somehow\n"
	"\n"
	"@type  model: L{ConditionalDistribution<models.ConditionalDistribution>}\n"
	"@param model: a conditional distribution such as an L{MCGSM<models.MCGSM>}\n"
	"\n"
	"@type  input_mask: C{ndarray}\n"
	"@param input_mask: a Boolean array describing the input pixels\n"
	"\n"
	"@type  output_mask: C{ndarray}\n"
	"@param output_mask: a Boolean array describing the output pixels\n"
	"\n"
	"@type  fmask: C{ndarray}\n"
	"@param fmask: a Boolean array describing the missing pixels\n"
	"\n"
	"@type  preconditioner: L{Preconditioner<transforms.Preconditioner>}\n"
	"@param preconditioner: transforms the input before feeding it into the model\n"
	"\n"
	"@type  num_iter: C{int}\n"
	"@param num_iter: number of iterations of replacing all pixels\n"
	"\n"
	"@type  num_steps: C{int}\n"
	"@param num_steps: number of Metropolis steps per pixel and iteration\n"
	"\n"
	"@rtype: C{ndarray}\n"
	"@return: an image with the missing pixels replaced";

PyObject* fill_in_image(PyObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"img", "model", "input_mask", "output_mask", "fmask", "preconditioner", "num_iter", "num_steps", 0};

	PyObject* img;
	PyObject* modelObj;
	PyObject* input_mask;
	PyObject* output_mask;
	PyObject* fmask;
	PyObject* preconditionerObj = 0;
	int num_iter = 10;
	int num_steps = 100;

	if(!PyArg_ParseTupleAndKeywords(args, kwds, "OO!OOO|O!ii", const_cast<char**>(kwlist),
		&img, &CD_type, &modelObj, &input_mask, &output_mask, &fmask,
		&Preconditioner_type, &preconditionerObj, &num_iter, &num_steps))
		return 0;

	const ConditionalDistribution& model = *reinterpret_cast<CDObject*>(modelObj)->cd;

	Preconditioner* preconditioner = preconditionerObj ?
		reinterpret_cast<PreconditionerObject*>(preconditionerObj)->preconditioner : 0;

	// make sure data is stored in NumPy array
	img = PyArray_FROM_OTF(img, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
	input_mask = PyArray_FROM_OTF(input_mask, NPY_BOOL, NPY_F_CONTIGUOUS | NPY_ALIGNED);
	output_mask = PyArray_FROM_OTF(output_mask, NPY_BOOL, NPY_F_CONTIGUOUS | NPY_ALIGNED);
	fmask = PyArray_FROM_OTF(fmask, NPY_BOOL, NPY_F_CONTIGUOUS | NPY_ALIGNED);

	if(!img) {
		Py_XDECREF(input_mask);
		Py_XDECREF(output_mask);
		Py_XDECREF(fmask);
		PyErr_SetString(PyExc_TypeError, "The image has to be given as an array.");
		return 0;
	}

	if(!input_mask || !output_mask || !fmask) {
		Py_DECREF(img);
		Py_XDECREF(input_mask);
		Py_XDECREF(output_mask);
		Py_XDECREF(fmask);
		PyErr_SetString(PyExc_TypeError, "Masks have to be given as Boolean arrays.");
		return 0;
	}

	try {
		PyObject* imgSample;
		if(PyArray_NDIM(img) > 2) {
			Py_DECREF(img);
			Py_DECREF(input_mask);
			Py_DECREF(output_mask);
			Py_DECREF(fmask);
			PyErr_SetString(PyExc_NotImplementedError, "Filling-in currently only works with grayscale images.");
			return 0;
		} else {
			// single-channel image and single-channel masks
			imgSample = PyArray_FromMatrixXd(
				fillInImage(
					PyArray_ToMatrixXd(img),
					model,
					PyArray_ToMatrixXb(input_mask),
					PyArray_ToMatrixXb(output_mask),
					PyArray_ToMatrixXb(fmask),
					preconditioner,
					num_iter,
					num_steps));
		}

		Py_DECREF(img);
		Py_DECREF(input_mask);
		Py_DECREF(output_mask);
		Py_DECREF(fmask);

		return imgSample;
	} catch(Exception& exception) {
		Py_DECREF(img);
		Py_DECREF(input_mask);
		Py_DECREF(output_mask);
		Py_DECREF(fmask);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	} catch(bad_alloc&) {
		Py_DECREF(img);
		Py_DECREF(input_mask);
		Py_DECREF(output_mask);
		PyErr_SetString(PyExc_RuntimeError, "Could not allocate memory.");
		return 0;
	}

	Py_DECREF(img);
	Py_DECREF(input_mask);
	Py_DECREF(output_mask);
	return 0;
}



PyObject* fill_in_image_map(PyObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"img", "model", "input_mask", "output_mask", "fmask", "preconditioner", "num_iter", "patch_size", 0};

	PyObject* img;
	PyObject* modelObj;
	PyObject* input_mask;
	PyObject* output_mask;
	PyObject* fmask;
	PyObject* preconditionerObj = 0;
	int num_iter = 10;
	int patch_size = 20;

	if(!PyArg_ParseTupleAndKeywords(args, kwds, "OO!OOO|O!ii", const_cast<char**>(kwlist),
		&img, &CD_type, &modelObj, &input_mask, &output_mask, &fmask,
		&Preconditioner_type, &preconditionerObj, &num_iter, &patch_size))
		return 0;

	const ConditionalDistribution& model = *reinterpret_cast<CDObject*>(modelObj)->cd;

	Preconditioner* preconditioner = preconditionerObj ?
		reinterpret_cast<PreconditionerObject*>(preconditionerObj)->preconditioner : 0;

	// make sure data is stored in NumPy array
	img = PyArray_FROM_OTF(img, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
	input_mask = PyArray_FROM_OTF(input_mask, NPY_BOOL, NPY_F_CONTIGUOUS | NPY_ALIGNED);
	output_mask = PyArray_FROM_OTF(output_mask, NPY_BOOL, NPY_F_CONTIGUOUS | NPY_ALIGNED);
	fmask = PyArray_FROM_OTF(fmask, NPY_BOOL, NPY_F_CONTIGUOUS | NPY_ALIGNED);

	if(!img) {
		Py_XDECREF(input_mask);
		Py_XDECREF(output_mask);
		Py_XDECREF(fmask);
		PyErr_SetString(PyExc_TypeError, "The image has to be given as an array.");
		return 0;
	}

	if(!input_mask || !output_mask || !fmask) {
		Py_DECREF(img);
		Py_XDECREF(input_mask);
		Py_XDECREF(output_mask);
		Py_XDECREF(fmask);
		PyErr_SetString(PyExc_TypeError, "Masks have to be given as Boolean arrays.");
		return 0;
	}

	try {
		PyObject* imgMAP;
		if(PyArray_NDIM(img) > 2) {
			Py_DECREF(img);
			Py_DECREF(input_mask);
			Py_DECREF(output_mask);
			Py_DECREF(fmask);
			PyErr_SetString(PyExc_NotImplementedError, "Filling-in currently only works with grayscale images.");
			return 0;
		} else {
			// single-channel image and single-channel masks
			imgMAP = PyArray_FromMatrixXd(
				fillInImageMAP(
					PyArray_ToMatrixXd(img),
					model,
					PyArray_ToMatrixXb(input_mask),
					PyArray_ToMatrixXb(output_mask),
					PyArray_ToMatrixXb(fmask),
					preconditioner,
					num_iter,
					patch_size));
		}

		Py_DECREF(img);
		Py_DECREF(input_mask);
		Py_DECREF(output_mask);
		Py_DECREF(fmask);

		return imgMAP;

	} catch(Exception& exception) {
		Py_DECREF(img);
		Py_DECREF(input_mask);
		Py_DECREF(output_mask);
		Py_DECREF(fmask);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	return 0;
}



const char* extract_windows_doc =
	"extract_windows(time_series, window_length)\n"
	"\n"
	"Extract windows from a given time series.\n"
	"\n"
	"This method can be used to, for example, extract stimulus and spike windows\n"
	"before training a neuron model such as the L{STM<models.STM>}.\n"
	"\n"
	"\t>>> spikes = extract_windows(spike_train, 20)\n"
	"\t>>> stimuli = extract_windows(stimulus, 20)\n"
	"\n"
	"The last row of C{spikes} contains the most recent bin of each window.\n"
	"We can use\n"
	"\n"
	"\t>>> inputs = vstack([stimuli, spikes[:-1]])\n"
	"\t>>> outputs = spikes[-1]\n"
	"\n"
	"to generate the inputs and outputs to a neuron model.\n"
	"\n"
	"\t>>> stm = STM(20, 50, 3, 10)\n"
	"\t>>> stm.train(inputs, outputs)\n"
	"\n"
	"@type  time_series: C{ndarray}\n"
	"@param time_series: an NxT array representing an N-dimensional time series of length T\n"
	"\n"
	"@type  window_length: C{int}\n"
	"@param window_length: number of bins of the extracted windows\n"
	"\n"
	"@rtype: C{tuple}\n"
	"@return: all possible overlapping windows of the time series";

PyObject* extract_windows(PyObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"time_series", "window_length", 0};

	PyObject* time_series;
	int window_length;

	if(!PyArg_ParseTupleAndKeywords(args, kwds, "Oi", const_cast<char**>(kwlist),
		&time_series, &window_length))
		return 0;

	time_series = PyArray_FROM_OTF(time_series, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

	if(!time_series) {
		PyErr_SetString(PyExc_TypeError, "time_series should be of type `ndarray`.");
		return 0;
	}

	try {
		PyObject* result = PyArray_FromMatrixXd(
			extractWindows(PyArray_ToMatrixXd(time_series), window_length));

		Py_DECREF(time_series);

		return result;
	} catch(Exception& exception) {
		Py_DECREF(time_series);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	} catch(bad_alloc&) {
		Py_DECREF(time_series);
		PyErr_SetString(PyExc_RuntimeError, "Could not allocate memory.");
		return 0;
	}


	Py_DECREF(time_series);

	return 0;
}



const char* sample_spike_train_doc =
	"sample_spike_train(stimuli, model, spike_history=0, preconditioner=None)\n"
	"\n"
	"Generate a spike train using a given model and a sequence of stimuli.\n"
	"\n"
	"If the model's output depends on the spike history, the length of the\n"
	"spike history taken into account can be specified using C{spike_history}.\n"
	"If C{spike_history} is positive, the model at time $t$ will receive the\n"
	"stimulus at time $t$ concatenated with C{spike_history} bins of the\n"
	"spike train preceding. The last entry of the input to the model is the\n"
	"most recent bin of the spike train.\n"
	"\n"
	"If C{preconditioner} is specified, the spike history is first transformed\n"
	"before appending it to the stimulus.\n"
	"\n"
	"@type  stimuli: C{ndarray}\n"
	"@param stimuli: each column represents a (possibly preprocessed) stimulus window\n"
	"\n"
	"@type  model: L{ConditionalDistribution<models.ConditionalDistribution>}\n"
	"@param model: a conditional distribution such as an L{STM<models.STM>}\n"
	"\n"
	"@type  spike_history: C{int}\n"
	"@param spike_history: number of bins used as input to the model\n"
	"\n"
	"@type  preconditioner: L{Preconditioner<transforms.Preconditioner>}\n"
	"@param preconditioner: transforms the spike history before appending it to the input\n"
	"\n"
	"@rtype: C{ndarray}\n"
	"@return: sampled spike train";

PyObject* sample_spike_train(PyObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {
		"stimulus", "model", "spike_history", "preconditioner", 0};

	PyObject* stimulus;
	PyObject* modelObj;
	int spike_history = 0;
	PyObject* preconditionerObj = 0;

	if(!PyArg_ParseTupleAndKeywords(args, kwds, "OO!|iO!", const_cast<char**>(kwlist),
		&stimulus,
		&CD_type, &modelObj,
		&spike_history,
		&Preconditioner_type, &preconditionerObj))
		return 0;

	if(preconditionerObj == Py_None)
		preconditionerObj = 0;

	stimulus = PyArray_FROM_OTF(stimulus, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);

	if(!stimulus) {
		PyErr_SetString(PyExc_TypeError, "stimulus should be of type `ndarray`.");
		return 0;
	}

	const ConditionalDistribution& model = *reinterpret_cast<CDObject*>(modelObj)->cd;

	Preconditioner* preconditioner = preconditionerObj ?
		reinterpret_cast<PreconditionerObject*>(preconditionerObj)->preconditioner : 0;

	try {
		ArrayXXd spikeTrain = sampleSpikeTrain(
			PyArray_ToMatrixXd(stimulus),
			model,
			spike_history,
			preconditioner);

		Py_DECREF(stimulus);
		
		return PyArray_FromMatrixXd(spikeTrain);
	} catch(Exception& exception) {
		Py_DECREF(stimulus);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	} catch(bad_alloc&) {
		Py_DECREF(stimulus);
		PyErr_SetString(PyExc_RuntimeError, "Could not allocate memory.");
		return 0;
	}

	Py_DECREF(stimulus);

	return 0;
}
