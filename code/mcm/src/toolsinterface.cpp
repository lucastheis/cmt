#include "toolsinterface.h"
#include "mcgsminterface.h"
#include "preconditionerinterface.h"
#include "conditionaldistributioninterface.h"
#include "exception.h"
#include "utils.h"
#include "tools.h"

#include <set>
using std::set;

#include <iostream>

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
	} catch(Exception exception) {
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}
	
	return 0;
}



const char* generate_data_from_image_doc =
	"generate_data_from_image(img, xmask, ymask, num_samples)\n"
	"\n"
	"Uniformly samples inputs and outputs for conditional models from images.\n"
	"\n"
	"@type  img: C{ndarray}\n"
	"@param img: an array representing a grayscale or color image\n"
	"\n"
	"@type  xmask: C{ndarray}\n"
	"@param xmask: a Boolean array describing the input pixels\n"
	"\n"
	"@type  ymask: C{ndarray}\n"
	"@param ymask: a Boolean array describing the output pixels\n"
	"\n"
	"@type  num_samples: C{int}\n"
	"@param num_samples: the number of generated input/output pairs\n"
	"\n"
	"@rtype: C{tuple}\n"
	"@return: the input and output vectors stored in columns";

PyObject* generate_data_from_image(PyObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"img", "xmask", "ymask", "num_samples", 0};

	PyObject* img;
	PyObject* xmask;
	PyObject* ymask;
	int num_samples;

	if(!PyArg_ParseTupleAndKeywords(args, kwds, "OOOi", const_cast<char**>(kwlist),
		&img, &xmask, &ymask, &num_samples))
		return 0;

	// make sure data is stored in NumPy array
	img = PyArray_FROM_OTF(img, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
	xmask = PyArray_FROM_OTF(xmask, NPY_BOOL, NPY_F_CONTIGUOUS | NPY_ALIGNED);
	ymask = PyArray_FROM_OTF(ymask, NPY_BOOL, NPY_F_CONTIGUOUS | NPY_ALIGNED);

	if(!img) {
		PyErr_SetString(PyExc_TypeError, "The initial image has to be given as an array.");
		return 0;
	}

	if(!xmask || !ymask) {
		Py_DECREF(img);
		Py_XDECREF(xmask);
		Py_XDECREF(ymask);
		PyErr_SetString(PyExc_TypeError, "Masks have to be given as Boolean arrays.");
		return 0;
	}

	try {
		pair<ArrayXXd, ArrayXXd> dataPair;
		
		if(PyArray_NDIM(img) > 2)
			if(PyArray_NDIM(xmask) > 2 && PyArray_NDIM(ymask) > 2)
				// multi-channel image and multi-channel masks
				dataPair = generateDataFromImage(
					PyArray_ToArraysXXd(img),
					PyArray_ToArraysXXb(xmask),
					PyArray_ToArraysXXb(ymask),
					num_samples);
			else
				// multi-channel image and single-channel masks
				dataPair = generateDataFromImage(
					PyArray_ToArraysXXd(img),
					PyArray_ToMatrixXb(xmask),
					PyArray_ToMatrixXb(ymask),
					num_samples);
		else
			// single-channel image and single-channel masks
			dataPair = generateDataFromImage(
				PyArray_ToMatrixXd(img),
				PyArray_ToMatrixXb(xmask),
				PyArray_ToMatrixXb(ymask),
				num_samples);

		PyObject* xvalues = PyArray_FromMatrixXd(dataPair.first);
		PyObject* yvalues = PyArray_FromMatrixXd(dataPair.second);

		PyObject* data = Py_BuildValue("(OO)",
			xvalues,
			yvalues);

		Py_DECREF(img);
		Py_DECREF(xvalues);
		Py_DECREF(yvalues);
		Py_DECREF(xmask);
		Py_DECREF(ymask);

		return data;

	} catch(Exception exception) {
		Py_DECREF(img);
		Py_DECREF(xmask);
		Py_DECREF(ymask);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	return 0;
}



const char* generate_data_from_video_doc =
	"generate_data_from_video(video, xmask, ymask, num_samples)\n"
	"\n"
	"Uniformly samples inputs and outputs for conditional models from videos.\n"
	"\n"
	"@type  video: C{ndarray}\n"
	"@param video: a three-dimensional array representing the video\n"
	"\n"
	"@type  xmask: C{ndarray}\n"
	"@param xmask: a three-dimensioal Boolean array describing the input pixels\n"
	"\n"
	"@type  ymask: C{ndarray}\n"
	"@param ymask: a three-dimensioal Boolean array describing the output pixels\n"
	"\n"
	"@type  num_samples: C{int}\n"
	"@param num_samples: the number of generated input/output pairs\n"
	"\n"
	"@rtype: C{tuple}\n"
	"@return: the input and output vectors stored in columns";

PyObject* generate_data_from_video(PyObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"video", "xmask", "ymask", "num_samples", 0};

	PyObject* video;
	PyObject* xmask;
	PyObject* ymask;
	int num_samples;

	if(!PyArg_ParseTupleAndKeywords(args, kwds, "OOOi", const_cast<char**>(kwlist),
		&video, &xmask, &ymask, &num_samples))
		return 0;

	// make sure data is stored in NumPy array
	video = PyArray_FROM_OTF(video, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
	xmask = PyArray_FROM_OTF(xmask, NPY_BOOL, NPY_F_CONTIGUOUS | NPY_ALIGNED);
	ymask = PyArray_FROM_OTF(ymask, NPY_BOOL, NPY_F_CONTIGUOUS | NPY_ALIGNED);

	if(!video) {
		PyErr_SetString(PyExc_TypeError, "The initial video has to be given as an array.");
		return 0;
	}

	if(!xmask || !ymask) {
		Py_DECREF(video);
		Py_XDECREF(xmask);
		Py_XDECREF(ymask);
		PyErr_SetString(PyExc_TypeError, "Masks have to be given as Boolean arrays.");
		return 0;
	}

	try {
		pair<ArrayXXd, ArrayXXd> dataPair;
		
		dataPair = generateDataFromVideo(
			PyArray_ToArraysXXd(video),
			PyArray_ToArraysXXb(xmask),
			PyArray_ToArraysXXb(ymask),
			num_samples);

		PyObject* xvalues = PyArray_FromMatrixXd(dataPair.first);
		PyObject* yvalues = PyArray_FromMatrixXd(dataPair.second);

		PyObject* data = Py_BuildValue("(OO)",
			xvalues,
			yvalues);

		Py_DECREF(video);
		Py_DECREF(xvalues);
		Py_DECREF(yvalues);
		Py_DECREF(xmask);
		Py_DECREF(ymask);

		return data;

	} catch(Exception exception) {
		Py_DECREF(video);
		Py_DECREF(xmask);
		Py_DECREF(ymask);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	return 0;
}


const char* sample_image_doc =
	"sample_image(img, model, xmask, ymask, preconditioner=None)\n"
	"\n"
	"Samples an image given a conditional distribution. The initial image passed to\n"
	"this function is used to initialize the boundaries and is also used to determine\n"
	"the size and length of the image to be generated.\n"
	"\n"
	"@type  img: C{ndarray}\n"
	"@param img: initialization of image\n"
	"\n"
	"@type  model: L{ConditionalDistribution}\n"
	"@param model: a conditional distribution such as an L{MCGSM}\n"
	"\n"
	"@type  xmask: C{ndarray}\n"
	"@param xmask: a Boolean array describing the input pixels\n"
	"\n"
	"@type  ymask: C{ndarray}\n"
	"@param ymask: a Boolean array describing the output pixels\n"
	"\n"
	"@type  preconditioner: L{Preconditioner}\n"
	"@param preconditioner: transforms the input before feeding it into the model\n"
	"\n"
	"@rtype: C{ndarray}\n"
	"@return: the sampled image";

PyObject* sample_image(PyObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"img", "model", "xmask", "ymask", "preconditioner", 0};

	PyObject* img;
	PyObject* modelObj;
	PyObject* xmask;
	PyObject* ymask;
	PyObject* preconditionerObj = 0;

	if(!PyArg_ParseTupleAndKeywords(args, kwds, "OO!OO|O!", const_cast<char**>(kwlist),
		&img, &CD_type, &modelObj, &xmask, &ymask, &Preconditioner_type, &preconditionerObj))
		return 0;

	const ConditionalDistribution& model = *reinterpret_cast<CDObject*>(modelObj)->cd;

	Preconditioner* preconditioner = preconditionerObj ?
		reinterpret_cast<PreconditionerObject*>(preconditionerObj)->preconditioner : 0;

	// make sure data is stored in NumPy array
	img = PyArray_FROM_OTF(img, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
	xmask = PyArray_FROM_OTF(xmask, NPY_BOOL, NPY_F_CONTIGUOUS | NPY_ALIGNED);
	ymask = PyArray_FROM_OTF(ymask, NPY_BOOL, NPY_F_CONTIGUOUS | NPY_ALIGNED);

	if(!img) {
		Py_XDECREF(xmask);
		Py_XDECREF(ymask);
		PyErr_SetString(PyExc_TypeError, "The initial image has to be given as an array.");
		return 0;
	}

	if(!xmask || !ymask) {
		Py_DECREF(img);
		Py_XDECREF(xmask);
		Py_XDECREF(ymask);
		PyErr_SetString(PyExc_TypeError, "Masks have to be given as Boolean arrays.");
		return 0;
	}

	try {
		PyObject* imgSample;
		if(PyArray_NDIM(img) > 2) {
			if(PyArray_NDIM(xmask) > 2 && PyArray_NDIM(ymask) > 2) {
				// multi-channel image and multi-channel masks
				imgSample = PyArray_FromArraysXXd(
					sampleImage(
						PyArray_ToArraysXXd(img),
						model,
						PyArray_ToArraysXXb(xmask),
						PyArray_ToArraysXXb(ymask),
						preconditioner));
			} else {
				// multi-channel image and single-channel masks
				imgSample = PyArray_FromArraysXXd(
					sampleImage(
						PyArray_ToArraysXXd(img),
						model,
						PyArray_ToMatrixXb(xmask),
						PyArray_ToMatrixXb(ymask),
						preconditioner));
			}
		} else {
			// single-channel image and single-channel masks
			imgSample = PyArray_FromMatrixXd(
				sampleImage(
					PyArray_ToMatrixXd(img),
					model,
					PyArray_ToMatrixXb(xmask),
					PyArray_ToMatrixXb(ymask),
					preconditioner));
		}

		Py_DECREF(img);
		Py_DECREF(xmask);
		Py_DECREF(ymask);

		return imgSample;

	} catch(Exception exception) {
		Py_DECREF(img);
		Py_DECREF(xmask);
		Py_DECREF(ymask);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	return 0;
}



const char* sample_video_doc =
	"sample_video(video, model, xmask, ymask, preconditioner=None)\n"
	"\n"
	"Samples a video given a conditional distribution. The initial video passed to\n"
	"this function is used to initialize the boundaries and is also used to determine\n"
	"the size and length of the video to be generated.\n"
	"\n"
	"@type  video: C{ndarray}\n"
	"@param video: initialization of video\n"
	"\n"
	"@type  model: L{ConditionalDistribution}\n"
	"@param model: a conditional distribution such as an L{MCGSM}\n"
	"\n"
	"@type  xmask: C{ndarray}\n"
	"@param xmask: a three-dimensional Boolean array describing the input pixels\n"
	"\n"
	"@type  ymask: C{ndarray}\n"
	"@param ymask: a three-dimensional Boolean array describing the output pixels\n"
	"\n"
	"@type  preconditioner: L{Preconditioner}\n"
	"@param preconditioner: transforms the input before feeding it into the model\n"
	"\n"
	"@rtype: C{ndarray}\n"
	"@return: the sampled video";

PyObject* sample_video(PyObject* self, PyObject* args, PyObject* kwds) {
	const char* kwlist[] = {"video", "model", "xmask", "ymask", "preconditioner", 0};

	PyObject* video;
	PyObject* modelObj;
	PyObject* xmask;
	PyObject* ymask;
	PyObject* preconditionerObj = 0;

	if(!PyArg_ParseTupleAndKeywords(args, kwds, "OO!OO|O!", const_cast<char**>(kwlist),
		&video, &CD_type, &modelObj, &xmask, &ymask, &Preconditioner_type, &preconditionerObj))
		return 0;

	const ConditionalDistribution& model = *reinterpret_cast<CDObject*>(modelObj)->cd;

	Preconditioner* preconditioner = preconditionerObj ?
		reinterpret_cast<PreconditionerObject*>(preconditionerObj)->preconditioner : 0;

	// make sure data is stored in NumPy array
	video = PyArray_FROM_OTF(video, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
	xmask = PyArray_FROM_OTF(xmask, NPY_BOOL, NPY_F_CONTIGUOUS | NPY_ALIGNED);
	ymask = PyArray_FROM_OTF(ymask, NPY_BOOL, NPY_F_CONTIGUOUS | NPY_ALIGNED);

	if(!video) {
		Py_XDECREF(xmask);
		Py_XDECREF(ymask);
		PyErr_SetString(PyExc_TypeError, "The initial video has to be given as an array.");
		return 0;
	}

	if(!xmask || !ymask) {
		Py_DECREF(video);
		Py_XDECREF(xmask);
		Py_XDECREF(ymask);
		PyErr_SetString(PyExc_TypeError, "Masks have to be given as Boolean arrays.");
		return 0;
	}

	try {
		PyObject* videoSample;

		videoSample = PyArray_FromArraysXXd(
			sampleVideo(
				PyArray_ToArraysXXd(video),
				model,
				PyArray_ToArraysXXb(xmask),
				PyArray_ToArraysXXb(ymask),
				preconditioner));

		Py_DECREF(video);
		Py_DECREF(xmask);
		Py_DECREF(ymask);

		return videoSample;

	} catch(Exception exception) {
		Py_DECREF(video);
		Py_DECREF(xmask);
		Py_DECREF(ymask);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	return 0;
}



const char* fill_in_image_doc =
	"fill_in_image(img, model, xmask, ymask, fmask, preconditioner=None, num_iter=10, num_steps=100)\n"
	"\n"
	"Samples pixels from an image conditioned on all other pixels.\n"
	"\n"
	"@type  img: C{ndarray}\n"
	"@param img: the image with the missing pixels initialized somehow\n"
	"\n"
	"@type  model: L{ConditionalDistribution}\n"
	"@param model: a conditional distribution such as an L{MCGSM}\n"
	"\n"
	"@type  xmask: C{ndarray}\n"
	"@param xmask: a Boolean array describing the input pixels\n"
	"\n"
	"@type  ymask: C{ndarray}\n"
	"@param ymask: a Boolean array describing the output pixels\n"
	"\n"
	"@type  fmask: C{ndarray}\n"
	"@param fmask: a Boolean array describing the missing pixels\n"
	"\n"
	"@type  preconditioner: L{Preconditioner}\n"
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
	const char* kwlist[] = {"img", "model", "xmask", "ymask", "fmask", "preconditioner", "num_iter", "num_steps", 0};

	PyObject* img;
	PyObject* modelObj;
	PyObject* xmask;
	PyObject* ymask;
	PyObject* fmask;
	PyObject* preconditionerObj = 0;
	int num_iter = 10;
	int num_steps = 100;

	if(!PyArg_ParseTupleAndKeywords(args, kwds, "OO!OOO|O!ii", const_cast<char**>(kwlist),
		&img, &CD_type, &modelObj, &xmask, &ymask, &fmask,
		&Preconditioner_type, &preconditionerObj, &num_iter, &num_steps))
		return 0;

	const ConditionalDistribution& model = *reinterpret_cast<CDObject*>(modelObj)->cd;

	Preconditioner* preconditioner = preconditionerObj ?
		reinterpret_cast<PreconditionerObject*>(preconditionerObj)->preconditioner : 0;

	// make sure data is stored in NumPy array
	img = PyArray_FROM_OTF(img, NPY_DOUBLE, NPY_F_CONTIGUOUS | NPY_ALIGNED);
	xmask = PyArray_FROM_OTF(xmask, NPY_BOOL, NPY_F_CONTIGUOUS | NPY_ALIGNED);
	ymask = PyArray_FROM_OTF(ymask, NPY_BOOL, NPY_F_CONTIGUOUS | NPY_ALIGNED);
	fmask = PyArray_FROM_OTF(fmask, NPY_BOOL, NPY_F_CONTIGUOUS | NPY_ALIGNED);

	if(!img) {
		Py_XDECREF(xmask);
		Py_XDECREF(ymask);
		Py_XDECREF(fmask);
		PyErr_SetString(PyExc_TypeError, "The image has to be given as an array.");
		return 0;
	}

	if(!xmask || !ymask || !fmask) {
		Py_DECREF(img);
		Py_XDECREF(xmask);
		Py_XDECREF(ymask);
		Py_XDECREF(fmask);
		PyErr_SetString(PyExc_TypeError, "Masks have to be given as Boolean arrays.");
		return 0;
	}

	try {
		PyObject* imgSample;
//		if(PyArray_NDIM(img) > 2) {
//			if(PyArray_NDIM(xmask) > 2 && PyArray_NDIM(ymask) > 2) {
//				// multi-channel image and multi-channel masks
//				imgSample = 
//					sampleImage(
//						PyArray_ToArraysXXd(img),
//						model,
//						PyArray_ToArraysXXb(xmask),
//						PyArray_ToArraysXXb(ymask),
//						preconditioner));
//			} else {
//				// multi-channel image and single-channel masks
//				imgSample = PyArray_FromArraysXXd(
//					sampleImage(
//						PyArray_ToArraysXXd(img),
//						model,
//						PyArray_ToMatrixXb(xmask),
//						PyArray_ToMatrixXb(ymask),
//						preconditioner));
//			}
//		} else {
			// single-channel image and single-channel masks
			imgSample = PyArray_FromMatrixXd(
				fillInImage(
					PyArray_ToMatrixXd(img),
					model,
					PyArray_ToMatrixXb(xmask),
					PyArray_ToMatrixXb(ymask),
					PyArray_ToMatrixXb(fmask),
					preconditioner,
					num_iter,
					num_steps));
//		}

		Py_DECREF(img);
		Py_DECREF(xmask);
		Py_DECREF(ymask);

		return imgSample;

	} catch(Exception exception) {
		Py_DECREF(img);
		Py_DECREF(xmask);
		Py_DECREF(ymask);
		PyErr_SetString(PyExc_RuntimeError, exception.message());
		return 0;
	}

	return 0;
}
