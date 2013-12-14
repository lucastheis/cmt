#include "lbfgs.h"
#include "utils.h"
#include "exception.h"

#include <algorithm>
using std::max;
using std::min;

#include <cstdio>
using std::rand;

#include <cmath>
using std::exp;
using std::log;

#include <cstring>
using std::memcpy;

#include <set>
using std::set;

#include <vector>
using std::vector;
using std::pair;
using std::make_pair;

#include <algorithm>
using std::find;
using std::copy;
using std::random_shuffle;

#include "tools.h"
using CMT::Tuple;
using CMT::Tuples;
using CMT::ConditionalDistribution;
using CMT::ArrayXXb;
using CMT::Preconditioner;
using CMT::extractFromImage;

#include "Eigen/Core"
using Eigen::Block;
using Eigen::Dynamic;
using Eigen::Array;
using Eigen::ArrayXd;
using Eigen::ArrayXXd;
using Eigen::ArrayXXi;
using Eigen::VectorXd;
using Eigen::Map;

#include <iostream>
#include <iomanip>
using std::cout;
using std::endl;

Tuples CMT::maskToIndices(const ArrayXXb& mask) {
	Tuples indices;

	for(int i = 0; i < mask.rows(); ++i)
		for(int j = 0; j < mask.cols(); ++j)
			if(mask(i, j))
				indices.push_back(make_pair(i, j));

	return indices;
}



pair<Tuples, Tuples> CMT::masksToIndices(
	const ArrayXXb& inputMask,
	const ArrayXXb& outputMask)
{
	if(inputMask.cols() != outputMask.cols() || inputMask.rows() != outputMask.rows())
		throw Exception("Input and output masks should be of the same size.");

	pair<Tuples, Tuples> indices;
	Tuples& inputIndices = indices.first;
	Tuples& outputIndices = indices.second;

	for(int i = 0; i < inputMask.rows(); ++i)
		for(int j = 0; j < inputMask.cols(); ++j) {
			if(inputMask(i, j))
				inputIndices.push_back(make_pair(i, j));
			if(outputMask(i, j))
				outputIndices.push_back(make_pair(i, j));
			if(inputMask(i, j) && outputMask(i, j))
				throw Exception("Input and output mask should not overlap.");
		}

	return indices;
}



VectorXd CMT::extractFromImage(const ArrayXXd& img, const Tuples& indices) {
	VectorXd pixels(indices.size());

	for(int i = 0; i < indices.size(); ++i)
		pixels[i] = img(indices[i].first, indices[i].second);

	return pixels;
}



pair<ArrayXXd, ArrayXXd> CMT::generateDataFromImage(
	const ArrayXXd& img,
	const ArrayXXb& inputMask,
	const ArrayXXb& outputMask)
{
	int w = img.cols() - inputMask.cols() + 1;
	int h = img.rows() - inputMask.rows() + 1;

	if(w < 1 || h < 1)
		throw Exception("Image not large enough for these masks.");

	// precompute indices of active pixels in masks
	pair<Tuples, Tuples> inOutIndices = masksToIndices(inputMask, outputMask);
	Tuples& inputIndices = inOutIndices.first;
	Tuples& outputIndices = inOutIndices.second;

	pair<ArrayXXd, ArrayXXd> data = make_pair(
		ArrayXXd(inputIndices.size(), w * h),
		ArrayXXd(outputIndices.size(), w * h));

	for(int k = 0, i = 0; i < h; ++i)
		for(int j = 0; j < w; ++j, ++k) {
			// extract input and output
			MatrixXd patch = img.block(i, j, inputMask.rows(), inputMask.cols());

			data.first.col(k) = extractFromImage(patch, inputIndices);
			data.second.col(k) = extractFromImage(patch, outputIndices);
		}

	return data;
}



pair<ArrayXXd, ArrayXXd> CMT::generateDataFromImage(
	const ArrayXXd& img,
	const ArrayXXb& inputMask,
	const ArrayXXb& outputMask,
	int numSamples)
{
	int w = img.cols() - inputMask.cols() + 1;
	int h = img.rows() - inputMask.rows() + 1;

	if(numSamples <= 0)
		return generateDataFromImage(img, inputMask, outputMask);

	if(numSamples > w * h)
		throw Exception("Image not large enough for this many samples.");

	// precompute indices of active pixels in masks
	pair<Tuples, Tuples> inOutIndices = masksToIndices(inputMask, outputMask);
	Tuples& inputIndices = inOutIndices.first;
	Tuples& outputIndices = inOutIndices.second;

	// sample random image locations
	set<int> indices = randomSelect(numSamples, w * h);

	// randomize order of indices
	vector<int> indicesRand(indices.size());
	copy(indices.begin(), indices.end(), indicesRand.begin());
	random_shuffle(indicesRand.begin(), indicesRand.end());

	// allocate memory
	pair<ArrayXXd, ArrayXXd> data = make_pair(
		ArrayXXd(inputIndices.size(), numSamples),
		ArrayXXd(outputIndices.size(), numSamples));

	int k = 0;

	for(vector<int>::iterator iter = indicesRand.begin(); iter != indicesRand.end(); ++iter, ++k) {
		// compute indices of image location
		int i = *iter / w;
		int j = *iter % w;

		// extract input and output
		MatrixXd patch = img.block(i, j, inputMask.rows(), inputMask.cols());

		data.first.col(k) = extractFromImage(patch, inputIndices);
		data.second.col(k) = extractFromImage(patch, outputIndices);
	}

	return data;
}



pair<ArrayXXd, ArrayXXd> CMT::generateDataFromImage(
	const vector<ArrayXXd>& img,
	const ArrayXXb& inputMask,
	const ArrayXXb& outputMask)
{
	int numChannels = img.size();

	if(!numChannels)
		throw Exception("Image should have at least one channel.");

	if(inputMask.cols() != outputMask.cols() || inputMask.rows() != outputMask.rows())
		throw Exception("Input and output masks should be of the same size.");

	int w = img[0].cols() - inputMask.cols() + 1;
	int h = img[0].rows() - inputMask.rows() + 1;

	// precompute indices of active pixels in masks
	pair<Tuples, Tuples> inOutIndices = masksToIndices(inputMask, outputMask);
	Tuples& inputIndices = inOutIndices.first;
	Tuples& outputIndices = inOutIndices.second;

	int numInputs = inputIndices.size();
	int numOutputs = outputIndices.size();

	pair<ArrayXXd, ArrayXXd> data = make_pair(
		ArrayXXd(numChannels * numInputs, w * h),
		ArrayXXd(numChannels * numOutputs, w * h));

	// extract inputs and outputs
	for(int k = 0, i = 0; i < h; ++i)
		for(int j = 0; j < w; ++j, ++k)
			for(int m = 0; m < numChannels; ++m) {
				MatrixXd patch = img[m].block(i, j, inputMask.rows(), inputMask.cols());
				data.first.block(m * numInputs, k, numInputs, 1) = extractFromImage(patch, inputIndices);
				data.second.block(m * numOutputs, k, numOutputs, 1) = extractFromImage(patch, outputIndices);
			}

	return data;
}



pair<ArrayXXd, ArrayXXd> CMT::generateDataFromImage(
	const vector<ArrayXXd>& img,
	const ArrayXXb& inputMask,
	const ArrayXXb& outputMask,
	int numSamples)
{
	if(numSamples <= 0)
		return generateDataFromImage(img, inputMask, outputMask);

	int numChannels = img.size();

	if(!numChannels)
		throw Exception("Image should have at least one channel.");

	if(inputMask.cols() != outputMask.cols() || inputMask.rows() != outputMask.rows())
		throw Exception("Input and output masks should be of the same size.");

	int w = img[0].cols() - inputMask.cols() + 1;
	int h = img[0].rows() - inputMask.rows() + 1;

	if(numSamples > w * h)
		throw Exception("Image not large enough for this many samples.");

	// precompute indices of active pixels in masks
	pair<Tuples, Tuples> inOutIndices = masksToIndices(inputMask, outputMask);
	Tuples& inputIndices = inOutIndices.first;
	Tuples& outputIndices = inOutIndices.second;

	// sample random image locations
	set<int> indices = randomSelect(numSamples, w * h);

	// randomize order of indices
	vector<int> indicesRand(indices.size());
	copy(indices.begin(), indices.end(), indicesRand.begin());
	random_shuffle(indicesRand.begin(), indicesRand.end());

	int numInputs = inputIndices.size();
	int numOutputs = outputIndices.size();

	pair<ArrayXXd, ArrayXXd> data = make_pair(
		ArrayXXd(numChannels * numInputs, numSamples),
		ArrayXXd(numChannels * numOutputs, numSamples));

	int k = 0;

	for(vector<int>::iterator iter = indicesRand.begin(); iter != indicesRand.end(); ++iter, ++k) {
		// compute indices of image location
		int i = *iter / w;
		int j = *iter % w;

		// extract input and output
		for(int m = 0; m < numChannels; ++m) {
			MatrixXd patch = img[m].block(i, j, inputMask.rows(), inputMask.cols());
			data.first.block(m * numInputs, k, numInputs, 1) = extractFromImage(patch, inputIndices);
			data.second.block(m * numOutputs, k, numOutputs, 1) = extractFromImage(patch, outputIndices);
		}
	}

	return data;
}



pair<ArrayXXd, ArrayXXd> CMT::generateDataFromImage(
	const vector<ArrayXXd>& img,
	const vector<ArrayXXb>& inputMask,
	const vector<ArrayXXb>& outputMask)
{
	int numChannels = img.size();

	if(!numChannels)
		throw Exception("Image should have at least one channel.");

	if(inputMask.size() != numChannels || outputMask.size() != numChannels)
		throw Exception("Image and masks need to have the same number of channels.");

	int w = img[0].cols() - inputMask[0].cols() + 1;
	int h = img[0].rows() - inputMask[0].rows() + 1;

	vector<Tuples> inputIndices;
	vector<Tuples> outputIndices;

	int numInputs = 0;
	int numOutputs = 0;

	// precompute indices of active pixels in masks
	for(int m = 0; m < numChannels; ++m) {
		if(inputMask[m].cols() != inputMask[0].cols() || inputMask[m].rows() != outputMask[0].rows())
			throw Exception("Input and output masks should be of the same size.");
		if(img[m].cols() != img[0].cols() || img[m].rows() != img[0].rows())
			throw Exception("All image channels should be of the same size.");

		// precompute indices of active pixels in masks
		pair<Tuples, Tuples> inOutIndices = masksToIndices(inputMask[m], outputMask[m]);
		inputIndices.push_back(inOutIndices.first);
		outputIndices.push_back(inOutIndices.second);

		numInputs += inputIndices[m].size();
		numOutputs += outputIndices[m].size();
	}

	pair<ArrayXXd, ArrayXXd> data = make_pair(
		ArrayXXd(numInputs, w * h),
		ArrayXXd(numOutputs, w * h));

	for(int k = 0, i = 0; i < h; ++i)
		for(int j = 0; j < w; ++j, ++k) {
			int offsetIn = 0;
			int offsetOut = 0;

			// extract input and output
			for(int m = 0; m < numChannels; ++m) {
				MatrixXd patch = img[m].block(i, j, inputMask[m].rows(), inputMask[m].cols());

				data.first.block(offsetIn, k, inputIndices[m].size(), 1) = extractFromImage(patch, inputIndices[m]);
				data.second.block(offsetOut, k, outputIndices[m].size(), 1) = extractFromImage(patch, outputIndices[m]);

				offsetIn += inputIndices[m].size();
				offsetOut += outputIndices[m].size();
			}
		}

	return data;
}



pair<ArrayXXd, ArrayXXd> CMT::generateDataFromImage(
	const vector<ArrayXXd>& img,
	const vector<ArrayXXb>& inputMask,
	const vector<ArrayXXb>& outputMask,
	int numSamples)
{
	if(numSamples <= 0)
		return generateDataFromImage(img, inputMask, outputMask);

	int numChannels = img.size();

	if(!numChannels)
		throw Exception("Image should have at least one channel.");

	if(inputMask.size() != numChannels || outputMask.size() != numChannels)
		throw Exception("Image and masks need to have the same number of channels.");

	int w = img[0].cols() - inputMask[0].cols() + 1;
	int h = img[0].rows() - inputMask[0].rows() + 1;

	if(numSamples > w * h)
		throw Exception("Image not large enough for this many samples.");

	vector<Tuples> inputIndices;
	vector<Tuples> outputIndices;

	int numInputs = 0;
	int numOutputs = 0;

	// precompute indices of active pixels in masks
	for(int m = 0; m < numChannels; ++m) {
		if(inputMask[m].cols() != inputMask[0].cols() || inputMask[m].rows() != outputMask[0].rows())
			throw Exception("Input and output masks should be of the same size.");
		if(img[m].cols() != img[0].cols() || img[m].rows() != img[0].rows())
			throw Exception("All image channels should be of the same size.");

		// precompute indices of active pixels in masks
		pair<Tuples, Tuples> inOutIndices = masksToIndices(inputMask[m], outputMask[m]);
		inputIndices.push_back(inOutIndices.first);
		outputIndices.push_back(inOutIndices.second);

		numInputs += inputIndices[m].size();
		numOutputs += outputIndices[m].size();
	}

	// sample random image locations
	set<int> indices = randomSelect(numSamples, w * h);

	// randomize order of indices
	vector<int> indicesRand(indices.size());
	copy(indices.begin(), indices.end(), indicesRand.begin());
	random_shuffle(indicesRand.begin(), indicesRand.end());

	pair<ArrayXXd, ArrayXXd> data = make_pair(
		ArrayXXd(numInputs, numSamples),
		ArrayXXd(numOutputs, numSamples));

	int k = 0;

	for(vector<int>::iterator iter = indicesRand.begin(); iter != indicesRand.end(); ++iter, ++k) {
		// compute indices of image location
		int i = *iter / w;
		int j = *iter % w;

		int offsetIn = 0;
		int offsetOut = 0;

		// extract input and output
		for(int m = 0; m < numChannels; ++m) {
			MatrixXd patch = img[m].block(i, j, inputMask[m].rows(), inputMask[m].cols());

			data.first.block(offsetIn, k, inputIndices[m].size(), 1) = extractFromImage(patch, inputIndices[m]);
			data.second.block(offsetOut, k, outputIndices[m].size(), 1) = extractFromImage(patch, outputIndices[m]);

			offsetIn += inputIndices[m].size();
			offsetOut += outputIndices[m].size();
		}
	}

	return data;
}



pair<ArrayXXd, ArrayXXd> CMT::generateDataFromVideo(
	const vector<ArrayXXd>& video,
	const vector<ArrayXXb>& inputMask,
	const vector<ArrayXXb>& outputMask)
{
	if(video.size() < inputMask.size())
		throw Exception("Video has less frames than masks.");

	if(inputMask.size() != outputMask.size())
		throw Exception("Masks need to have the same number of frames.");

	int w = video[0].cols() - inputMask[0].cols() + 1;
	int h = video[0].rows() - inputMask[0].rows() + 1;
	int l = video.size() - inputMask.size() + 1;

	vector<Tuples> inputIndices;
	vector<Tuples> outputIndices;

	int numInputs = 0;
	int numOutputs = 0;

	// precompute indices of active pixels in masks
	for(int m = 0; m < inputMask.size(); ++m) {
		if(inputMask[m].cols() != inputMask[0].cols() || inputMask[m].rows() != outputMask[0].rows())
			throw Exception("Input and output masks should be of the same size.");
		if(video[m].cols() != video[0].cols() || video[m].rows() != video[0].rows())
			throw Exception("All video frames should be of the same size.");

		pair<Tuples, Tuples> inOutIndices = masksToIndices(inputMask[m], outputMask[m]);
		inputIndices.push_back(inOutIndices.first);
		outputIndices.push_back(inOutIndices.second);

		numInputs += inputIndices[m].size();
		numOutputs += outputIndices[m].size();
	}

	pair<ArrayXXd, ArrayXXd> data = make_pair(
		ArrayXXd(numInputs, w * h * l),
		ArrayXXd(numOutputs, w * h * l));

	for(int k = 0, f = 0; f < l; ++f)
		for(int i = 0; i < h; ++i)
			for(int j = 0; j < w; ++j, ++k) {
				int offsetIn = 0;
				int offsetOut = 0;

				// extract input and output
				for(int m = 0; m < inputMask.size(); ++m) {
					MatrixXd patch = video[m + f].block(i, j, inputMask[m].rows(), inputMask[m].cols());

					data.first.block(offsetIn, k, inputIndices[m].size(), 1) = extractFromImage(patch, inputIndices[m]);
					data.second.block(offsetOut, k, outputIndices[m].size(), 1) = extractFromImage(patch, outputIndices[m]);

					offsetIn += inputIndices[m].size();
					offsetOut += outputIndices[m].size();
				}
			}

	return data;
}



pair<ArrayXXd, ArrayXXd> CMT::generateDataFromVideo(
	const vector<ArrayXXd>& video,
	const vector<ArrayXXb>& inputMask,
	const vector<ArrayXXb>& outputMask,
	int numSamples)
{
	if(numSamples <= 0)
		return generateDataFromVideo(video, inputMask, outputMask);

	if(video.size() < inputMask.size())
		throw Exception("Video has less frames than masks.");

	if(inputMask.size() != outputMask.size())
		throw Exception("Masks need to have the same number of frames.");

	int w = video[0].cols() - inputMask[0].cols() + 1;
	int h = video[0].rows() - inputMask[0].rows() + 1;
	int l = video.size() - inputMask.size() + 1;

	if(numSamples > w * h * l)
		throw Exception("Video not large enough for this many samples.");

	vector<Tuples> inputIndices;
	vector<Tuples> outputIndices;

	int numInputs = 0;
	int numOutputs = 0;

	// precompute indices of active pixels in masks
	for(int m = 0; m < inputMask.size(); ++m) {
		if(inputMask[m].cols() != inputMask[0].cols() || inputMask[m].rows() != outputMask[0].rows())
			throw Exception("Input and output masks should be of the same size.");
		if(video[m].cols() != video[0].cols() || video[m].rows() != video[0].rows())
			throw Exception("All video frames should be of the same size.");

		pair<Tuples, Tuples> inOutIndices = masksToIndices(inputMask[m], outputMask[m]);
		inputIndices.push_back(inOutIndices.first);
		outputIndices.push_back(inOutIndices.second);

		numInputs += inputIndices[m].size();
		numOutputs += outputIndices[m].size();
	}

	// sample random video locations
	set<int> indices = randomSelect(numSamples, w * h * l);

	// randomize order of indices
	vector<int> indicesRand(indices.size());
	copy(indices.begin(), indices.end(), indicesRand.begin());
	random_shuffle(indicesRand.begin(), indicesRand.end());

	pair<ArrayXXd, ArrayXXd> data = make_pair(
		ArrayXXd(numInputs, numSamples),
		ArrayXXd(numOutputs, numSamples));

	int k = 0;

	for(vector<int>::iterator iter = indicesRand.begin(); iter != indicesRand.end(); ++iter, ++k) {
		// compute indices of video location
		int f = *iter / (w * h);
		int r = *iter % (w * h);
		int i = r / w;
		int j = r % w;

		int offsetIn = 0;
		int offsetOut = 0;

		// extract input and output
		for(int m = 0; m < inputMask.size(); ++m) {
			MatrixXd patch = video[m + f].block(i, j, inputMask[m].rows(), inputMask[m].cols());

			data.first.block(offsetIn, k, inputIndices[m].size(), 1) = extractFromImage(patch, inputIndices[m]);
			data.second.block(offsetOut, k, outputIndices[m].size(), 1) = extractFromImage(patch, outputIndices[m]);

			offsetIn += inputIndices[m].size();
			offsetOut += outputIndices[m].size();
		}
	}

	return data;
}



ArrayXXd CMT::sampleImage(
	ArrayXXd img,
	const ConditionalDistribution& model,
	const ArrayXXb& inputMask,
	const ArrayXXb& outputMask,
	const Preconditioner* preconditioner)
{
	if(inputMask.cols() != outputMask.cols() || inputMask.rows() != outputMask.rows())
		throw Exception("Input and output masks should be of the same size.");

	Tuples inputIndices;
	Tuples outputIndices;

	// boundaries of output region
	int iMin = outputMask.rows();
	int iMax = 0;
	int jMin = outputMask.cols();
	int jMax = 0;

	// precompute indices of active pixels in masks
	for(int i = 0; i < inputMask.rows(); ++i)
		for(int j = 0; j < inputMask.cols(); ++j) {
			if(inputMask(i, j))
				inputIndices.push_back(make_pair(i, j));
			if(outputMask(i, j)) {
				outputIndices.push_back(make_pair(i, j));

				// update boundaries
				iMax = max(iMax, i);
				iMin = min(iMin, i);
				jMax = max(jMax, j);
				jMin = min(jMin, j);
			}
			if(inputMask(i, j) && outputMask(i, j))
				throw Exception("Input and output mask should not overlap.");
		}

	// width and height of output region
	int h = iMax - iMin + 1;
	int w = jMax - jMin + 1;

	if(w < 1)
		throw Exception("There needs to be at least one active pixel in the output mask.");

	if(outputIndices.size() != w * h)
		throw Exception("Unsupported output mask.");

	if(preconditioner) {
		if(inputIndices.size() != preconditioner->dimIn() || outputIndices.size() != preconditioner->dimOut())
			throw Exception("Preconditioner and masks are incompatible.");
		if(preconditioner->dimInPre() != model.dimIn() || preconditioner->dimOutPre() != model.dimOut())
			throw Exception("Model and preconditioner are incompatible.");
	} else {
		if(inputIndices.size() != model.dimIn() || outputIndices.size() != model.dimOut())
			throw Exception("Model and masks are incompatible.");
	}

	for(int i = 0; i + inputMask.rows() <= img.rows(); i += h)
		for(int j = 0; j + inputMask.cols() <= img.cols(); j += w) {
			// extract causal neighborhood
			VectorXd input = extractFromImage(
				img.block(i, j, inputMask.rows(), inputMask.cols()), inputIndices);

			VectorXd output;

			// sample output
			if(preconditioner) {
				input = preconditioner->operator()(input);
				output = preconditioner->inverse(input, model.sample(input)).second;
			} else {
				output = model.sample(input);
			}

			// replace pixels in image by output
			for(int k = 0; k < outputIndices.size(); ++k)
				img(i + outputIndices[k].first, j + outputIndices[k].second) = output[k];
		}

	return img;
}



vector<ArrayXXd> CMT::sampleImage(
	vector<ArrayXXd> img,
	const ConditionalDistribution& model,
	const ArrayXXb& inputMask,
	const ArrayXXb& outputMask,
	const Preconditioner* preconditioner)
{
	if(!img.size())
		throw Exception("Image should have at least one channel.");

	if(inputMask.cols() != outputMask.cols() || inputMask.rows() != outputMask.rows())
		throw Exception("Input and output masks should be of the same size.");

	Tuples inputIndices;
	Tuples outputIndices;

	// boundaries of output region
	int iMin = outputMask.rows();
	int iMax = 0;
	int jMin = outputMask.cols();
	int jMax = 0;

	// precompute indices of active pixels in masks
	for(int i = 0; i < inputMask.rows(); ++i)
		for(int j = 0; j < inputMask.cols(); ++j) {
			if(inputMask(i, j))
				inputIndices.push_back(make_pair(i, j));
			if(outputMask(i, j)) {
				outputIndices.push_back(make_pair(i, j));

				// update boundaries
				iMax = max(iMax, i);
				iMin = min(iMin, i);
				jMax = max(jMax, j);
				jMin = min(jMin, j);
			}

			if(inputMask(i, j) && outputMask(i, j))
				throw Exception("Input and output mask should not overlap.");
		}

	// width and height of output region
	int h = iMax - iMin + 1;
	int w = jMax - jMin + 1;

	if(w < 1)
		throw Exception("There needs to be at least one active pixel in the output mask.");

	int numInputs = inputIndices.size();
	int numOutputs = outputIndices.size();
	int numChannels = img.size();

	if(numOutputs != w * h)
		throw Exception("Unsupported output mask.");

	if(preconditioner) {
		if(numInputs * numChannels != preconditioner->dimIn() || numOutputs * numChannels != preconditioner->dimOut())
			throw Exception("Preconditioner and masks are incompatible.");
		if(preconditioner->dimInPre() != model.dimIn() || preconditioner->dimOutPre() != model.dimOut())
			throw Exception("Model and preconditioner are incompatible.");
	} else {
		if(numInputs * numChannels != model.dimIn() || numOutputs * numChannels != model.dimOut())
			throw Exception("Model and masks are incompatible.");
	}

	for(int i = 0; i + inputMask.rows() <= img[0].rows(); i += h)
		for(int j = 0; j + inputMask.cols() <= img[0].cols(); j += w) {
			VectorXd input(numInputs * numChannels);

			// extract causal neighborhood
			#pragma omp parallel for
			for(int m = 0; m < numChannels; ++m)
				input.segment(m * numInputs, numInputs) = extractFromImage(
					img[m].block(i, j, inputMask.rows(), inputMask.cols()), inputIndices);

			// sample output
			VectorXd output;
			
			if(preconditioner) {
				std::cout << "in: " << input.rows() << ", " << input.cols() << std::endl;
				input = preconditioner->operator()(input);
				output = model.sample(input);
				std::cout << "ou: " << output.rows() << ", " << output.cols() << std::endl;
				output = preconditioner->inverse(input, output).second;
			} else {
				output = model.sample(input);
			}

			// replace pixels in image by output
			#pragma omp parallel for
			for(int m = 0; m < numChannels; ++m)
				for(int k = 0; k < outputIndices.size(); ++k)
					img[m](i + outputIndices[k].first, j + outputIndices[k].second) = output[m * numOutputs + k];
		}

	return img;
}



vector<ArrayXXd> CMT::sampleImage(
	vector<ArrayXXd> img,
	const ConditionalDistribution& model,
	const vector<ArrayXXb>& inputMask,
	const vector<ArrayXXb>& outputMask,
	const Preconditioner* preconditioner)
{
	int numChannels = img.size();

	if(!numChannels)
		throw Exception("Image should have at least one channel.");

	if(inputMask.size() != numChannels || outputMask.size() != numChannels)
		throw Exception("Image and masks need to have the same number of channels.");

	vector<Tuples> inputIndices;
	vector<Tuples> outputIndices;

	// boundaries of output region
	int iMin = outputMask[0].rows();
	int iMax = 0;
	int jMin = outputMask[0].cols();
	int jMax = 0;

	int numInputs = 0;
	int numOutputs = 0;

	for(int m = 0; m < numChannels; ++m) {
		if(inputMask[m].cols() != outputMask[m].cols() || inputMask[m].rows() != outputMask[m].rows())
			throw Exception("Input and output masks should be of the same size.");
		if(inputMask[m].cols() != inputMask[0].cols() || inputMask[m].rows() != outputMask[0].rows())
			throw Exception("Input and output masks should be of the same size.");
		if(img[m].cols() != img[0].cols() || img[m].rows() != img[0].rows())
			throw Exception("All image channels should be of the same size.");

		inputIndices.push_back(Tuples());
		outputIndices.push_back(Tuples());

		// precompute indices of active pixels in masks
		for(int i = 0; i < inputMask[m].rows(); ++i)
			for(int j = 0; j < inputMask[m].cols(); ++j) {
				if(inputMask[m](i, j))
					inputIndices[m].push_back(make_pair(i, j));
				if(outputMask[m](i, j)) {
					outputIndices[m].push_back(make_pair(i, j));

					// update boundaries
					iMax = max(iMax, i);
					iMin = min(iMin, i);
					jMax = max(jMax, j);
					jMin = min(jMin, j);
				}

				if(inputMask[m](i, j) && outputMask[m](i, j))
					throw Exception("Input and output mask should not overlap.");
			}

		numInputs += inputIndices[m].size();
		numOutputs += outputIndices[m].size();
	}

	// width and height of output region
	int h = iMax - iMin + 1;
	int w = jMax - jMin + 1;

	if(w < 1)
		throw Exception("There needs to be at least one active pixel in the output mask.");

	if(numOutputs % (w * h))
		throw Exception("Unsupported output mask.");

	if(preconditioner) {
		if(numInputs != preconditioner->dimIn() || numOutputs != preconditioner->dimOut())
			throw Exception("Preconditioner and masks are incompatible.");
		if(preconditioner->dimInPre() != model.dimIn() || preconditioner->dimOutPre() != model.dimOut())
			throw Exception("Model and preconditioner are incompatible.");
	} else {
		if(numInputs != model.dimIn() || numOutputs != model.dimOut())
			throw Exception("Model and masks are incompatible.");
	}

	for(int i = 0; i + inputMask[0].rows() <= img[0].rows(); i += h)
		for(int j = 0; j + inputMask[0].cols() <= img[0].cols(); j += w) {
			VectorXd input(numInputs);

			// extract causal neighborhood
			for(int m = 0, offset = 0; m < numChannels; ++m) {
				input.segment(offset, inputIndices[m].size()) = extractFromImage(
					img[m].block(i, j, inputMask[m].rows(), inputMask[m].cols()), inputIndices[m]);
				offset += inputIndices[m].size();
			}

			// sample output
			VectorXd output;

			if(preconditioner) {
				input = preconditioner->operator()(input);
				output = preconditioner->inverse(input, model.sample(input)).second;
			} else {
				output = model.sample(input);
			}

			// replace pixels in image by model's output
			for(int m = 0, offset = 0; m < numChannels; ++m) {
				for(int k = 0; k < outputIndices[m].size(); ++k)
					img[m](i + outputIndices[m][k].first, j + outputIndices[m][k].second) = output[offset + k];
				offset += outputIndices[m].size();
			}
		}

	return img;
}



ArrayXXd CMT::sampleImageConditionally(
	ArrayXXd img,
	ArrayXXi labels,
	const MCGSM& model,
	const ArrayXXb& inputMask,
	const ArrayXXb& outputMask,
	const Preconditioner* preconditioner,
	int numIter,
	bool initialize)
{
	if(inputMask.cols() != outputMask.cols() || inputMask.rows() != outputMask.rows())
		throw Exception("Input and output masks should be of the same size.");

	Tuples inputIndices;
	Tuples outputIndices;

	// boundaries of output region
	int iMin = outputMask.rows();
	int iMax = 0;
	int jMin = outputMask.cols();
	int jMax = 0;

	// precompute indices of active pixels in masks
	for(int i = 0; i < inputMask.rows(); ++i)
		for(int j = 0; j < inputMask.cols(); ++j) {
			if(inputMask(i, j))
				inputIndices.push_back(make_pair(i, j));
			if(outputMask(i, j)) {
				outputIndices.push_back(make_pair(i, j));

				// update boundaries
				iMax = max(iMax, i);
				iMin = min(iMin, i);
				jMax = max(jMax, j);
				jMin = min(jMin, j);
			}
			if(inputMask(i, j) && outputMask(i, j))
				throw Exception("Input and output mask should not overlap.");
		}

	// width and height of output region
	int h = iMax - iMin + 1;
	int w = jMax - jMin + 1;

	if(w < 1)
		throw Exception("There needs to be at least one active pixel in the output mask.");

	// at the moment, only rectangular output regions are supported
	if(outputIndices.size() != w * h)
		throw Exception("Unsupported output mask.");

	if(preconditioner) {
		if(inputIndices.size() != preconditioner->dimIn() || outputIndices.size() != preconditioner->dimOut())
			throw Exception("Preconditioner and masks are incompatible.");
		if(preconditioner->dimInPre() != model.dimIn() || preconditioner->dimOutPre() != model.dimOut())
			throw Exception("Model and preconditioner are incompatible.");
	} else {
		if(inputIndices.size() != model.dimIn() || outputIndices.size() != model.dimOut())
			throw Exception("Model and masks are incompatible.");
	}

	// extract causal neighborhoods and corresponding output regions
	vector<vector<VectorXd> > inputs;
	vector<vector<VectorXd> > outputs;
	vector<vector<double> > logLik;
	vector<vector<double> > logPrb;

	for(int i = 0; i + inputMask.rows() <= img.rows(); ++i) {
		inputs.push_back(vector<VectorXd>());
		outputs.push_back(vector<VectorXd>());
		logLik.push_back(vector<double>());
		logPrb.push_back(vector<double>());
 
 		if(i % h != 0)
 			continue;

		for(int j = 0; j + inputMask.cols() <= img.cols(); ++j) {
 			if(j % w != 0) {
 				// create dummy variables
 				inputs[i].push_back(VectorXd());
 				outputs[i].push_back(VectorXd());
 				logLik[i].push_back(0.);
 				logPrb[i].push_back(0.);
 				continue;
 			}

			ArrayXXd patch = img.block(i, j, inputMask.rows(), inputMask.cols());
			inputs[i].push_back(extractFromImage(patch, inputIndices));
			outputs[i].push_back(extractFromImage(patch, outputIndices));

			Array<int, 1, Dynamic> label(1);
			label[0] = labels(i / h, j / w);

			if(preconditioner) {
				pair<ArrayXXd, ArrayXXd> data = preconditioner->operator()(inputs[i][j], outputs[i][j]);
				logLik[i].push_back(model.logLikelihood(data.first, data.second, label)[0]);
				logPrb[i].push_back(log(model.prior(data.first)(label[0], 0)));
			} else {
				logLik[i].push_back(model.logLikelihood(inputs[i][j], outputs[i][j], label)[0]);
				logPrb[i].push_back(log(model.prior(inputs[i][j])(label[0], 0)));
			}
		}
	}

	// precompute which neighborhoods have to be considered in acceptance probability
	Tuples offsets;
	for(int m = 0; m < inputIndices.size(); ++m)
		for(int n = 0; n < outputIndices.size(); ++n) {
			int di = outputIndices[n].first  - inputIndices[m].first;
			int dj = outputIndices[n].second - inputIndices[m].second;

			if(di % h == 0 && dj % w == 0)
				offsets.push_back(make_pair(di, dj));
		}

	// for each offset, compute indices of input pixels affected by Gibbs update
	vector<vector<int> > idxIn;
	vector<vector<int> > idxOut;

	for(int k = 0; k < offsets.size(); ++k) {
		// list of pixels in input at given offset which need to be updated
		idxIn.push_back(vector<int>());
		idxOut.push_back(vector<int>());

		for(int l = 0; l < outputIndices.size(); ++l) {
			// compute location of affected pixel in shifted input region
			int i = outputIndices[l].first - offsets[k].first;
			int j = outputIndices[l].second - offsets[k].second;

			Tuples::iterator it = find(inputIndices.begin(), inputIndices.end(), make_pair(i, j));

			if(it != inputIndices.end())
				idxIn[k].push_back(distance(inputIndices.begin(), it));
				idxOut[k].push_back(l);
		}
	}

	for(int iter = 0; iter < numIter; ++iter)
		// walk through image from top left to bottom right
		for(int i = 0; i + inputMask.rows() <= img.rows(); i += h)
			for(int j = 0; j + inputMask.cols() <= img.cols(); j += w) {
				Array<int, 1, Dynamic> label(1);
				label[0] = labels(i / h, j / w);

				// propose output
				VectorXd output;

				if(preconditioner) {
					VectorXd input = preconditioner->operator()(inputs[i][j]);
					output = preconditioner->inverse(input, model.sample(input, label)).second;
				} else {
					output = model.sample(inputs[i][j], label);
				}

				vector<VectorXd> inputsUpdated;
				vector<double> logLikUpdated;
				vector<double> logPrbUpdated;

				// compute acceptance probability
				double logAlpha = 0.;

				for(int k = 0; k < offsets.size(); ++k) {
					int m = i + offsets[k].first;
					int n = j + offsets[k].second;

					if(m >= inputs.size() || n >= inputs[m].size()) {
						inputsUpdated.push_back(VectorXd());
						logLikUpdated.push_back(0.);
						logPrbUpdated.push_back(0.);
						continue;
					}

					// replace affected pixels in causal neighborhood
					inputsUpdated.push_back(inputs[m][n]);
					for(int l = 0; l < idxIn[k].size(); ++l)
						inputsUpdated[k][idxIn[k][l]] = output[idxOut[k][l]];

					// update likelihoods and probabilities needed for computing alpha
					Array<int, 1, Dynamic> label(1);
					label[0] = labels(m / h, n / w);

					if(preconditioner) {
						pair<ArrayXXd, ArrayXXd> data = preconditioner->operator()(inputsUpdated[k], outputs[m][n]);

						logLikUpdated.push_back(model.logLikelihood(data.first, data.second, label)[0]);
						logAlpha += logLikUpdated[k] - logLik[m][n];

						logPrbUpdated.push_back(log(model.prior(data.first)(label[0], 0)));
						logAlpha += logPrbUpdated[k] - logPrb[m][n];
					} else {
						logLikUpdated.push_back(model.logLikelihood(inputsUpdated[k], outputs[m][n], label)[0]);
						logAlpha += logLikUpdated[k] - logLik[m][n];

						logPrbUpdated.push_back(log(model.prior(inputsUpdated[k])(label[0], 0)));
						logAlpha += logPrbUpdated[k] - logPrb[m][n];
					}
				}

				// accept/reject proposed output
				if((iter == 0 && initialize) || log(rand() / static_cast<double>(RAND_MAX)) < logAlpha) {
					// update inputs and outputs
					outputs[i][j] = output;

					for(int k = 0; k < offsets.size(); ++k) {
						int m = i + offsets[k].first;
						int n = j + offsets[k].second;

						if(m >= inputs.size() || n >= inputs[m].size())
							continue;

						inputs[m][n] = inputsUpdated[k];
						logLik[m][n] = logLikUpdated[k];
						logPrb[m][n] = logPrbUpdated[k];
					}
				}
			}

	// replace pixels by sampled pixels
	for(int i = 0; i + inputMask.rows() <= img.rows(); i += h)
		for(int j = 0; j + inputMask.cols() <= img.cols(); j += w)
			for(int k = 0; k < outputIndices.size(); ++k)
				img(i + outputIndices[k].first, j + outputIndices[k].second) = outputs[i][j][k];

	return img;
}



ArrayXXi CMT::sampleLabelsConditionally(
	ArrayXXd img,
	const MCGSM& model,
	const ArrayXXb& inputMask,
	const ArrayXXb& outputMask,
	const Preconditioner* preconditioner)
{
	if(inputMask.cols() != outputMask.cols() || inputMask.rows() != outputMask.rows())
		throw Exception("Input and output masks should be of the same size.");

	Tuples inputIndices;
	Tuples outputIndices;

	// boundaries of output region
	int iMin = outputMask.rows();
	int iMax = 0;
	int jMin = outputMask.cols();
	int jMax = 0;

	// precompute indices of active pixels in masks
	for(int i = 0; i < inputMask.rows(); ++i)
		for(int j = 0; j < inputMask.cols(); ++j) {
			if(inputMask(i, j))
				inputIndices.push_back(make_pair(i, j));
			if(outputMask(i, j)) {
				outputIndices.push_back(make_pair(i, j));

				// update boundaries
				iMax = max(iMax, i);
				iMin = min(iMin, i);
				jMax = max(jMax, j);
				jMin = min(jMin, j);
			}
			if(inputMask(i, j) && outputMask(i, j))
				throw Exception("Input and output mask should not overlap.");
		}

	// width and height of output region
	int h = iMax - iMin + 1;
	int w = jMax - jMin + 1;

	if(w < 1)
		throw Exception("There needs to be at least one active pixel in the output mask.");

	if(outputIndices.size() != w * h)
		throw Exception("Unsupported output mask.");

	if(preconditioner) {
		if(inputIndices.size() != preconditioner->dimIn() || outputIndices.size() != preconditioner->dimOut())
			throw Exception("Preconditioner and masks are incompatible.");
		if(preconditioner->dimInPre() != model.dimIn() || preconditioner->dimOutPre() != model.dimOut())
			throw Exception("Model and preconditioner are incompatible.");
	} else {
		if(inputIndices.size() != model.dimIn() || outputIndices.size() != model.dimOut())
			throw Exception("Model and masks are incompatible.");
	}

	ArrayXXi labels(
		(img.rows() - inputMask.rows()) / h + 1,
		(img.cols() - inputMask.cols()) / w + 1);

	labels.setConstant(-1);

	for(int i = 0; i + inputMask.rows() <= img.rows(); i += h) {
		for(int j = 0; j + inputMask.cols() <= img.cols(); j += w) {
			ArrayXXd patch = img.block(i, j, inputMask.rows(), inputMask.cols());

			ArrayXd input  = extractFromImage(patch, inputIndices);
			ArrayXd output = extractFromImage(patch, outputIndices);

			if(preconditioner) {
				pair<ArrayXXd, ArrayXXd> data = preconditioner->operator()(input, output);
				labels(i / h, j / w) = model.samplePosterior(data.first, data.second)[0];
			} else {
				labels(i / h, j / w) = model.samplePosterior(input, output)[0];
			}
		}
	}

	return labels;
}



vector<ArrayXXd> CMT::sampleVideo(
	vector<ArrayXXd> video,
	const ConditionalDistribution& model,
	const vector<ArrayXXb>& inputMask,
	const vector<ArrayXXb>& outputMask,
	const Preconditioner* preconditioner)
{
	if(video.size() < inputMask.size())
		return video;

	if(inputMask.size() != outputMask.size())
		throw Exception("Masks need to have the same number of frames.");

	vector<Tuples> inputIndices;
	vector<Tuples> outputIndices;

	// boundaries of output region
	int fMin = outputMask.size();
	int fMax = 0;
	int iMin = outputMask[0].rows();
	int iMax = 0;
	int jMin = outputMask[0].cols();
	int jMax = 0;

	int numInputs = 0;
	int numOutputs = 0;

	for(int m = 0; m < inputMask.size(); ++m) {
		if(inputMask[m].cols() != outputMask[m].cols() || inputMask[m].rows() != outputMask[m].rows())
			throw Exception("Input and output masks should be of the same size.");
		if(inputMask[m].cols() != inputMask[0].cols() || inputMask[m].rows() != outputMask[0].rows())
			throw Exception("Input and output masks should be of the same size.");
		if(video[m].cols() != video[0].cols() || video[m].rows() != video[0].cols())
			throw Exception("All video frames should be of the same size.");

		inputIndices.push_back(Tuples());
		outputIndices.push_back(Tuples());

		// precompute indices of active pixels in masks
		for(int i = 0; i < inputMask[m].rows(); ++i)
			for(int j = 0; j < inputMask[m].cols(); ++j) {
				if(inputMask[m](i, j))
					inputIndices[m].push_back(make_pair(i, j));
				if(outputMask[m](i, j)) {
					outputIndices[m].push_back(make_pair(i, j));

					// update boundaries
					fMax = max(fMax, m);
					fMin = min(fMin, m);
					iMax = max(iMax, i);
					iMin = min(iMin, i);
					jMax = max(jMax, j);
					jMin = min(jMin, j);
				}

				if(inputMask[m](i, j) && outputMask[m](i, j))
					throw Exception("Input and output mask should not overlap.");
			}

		numInputs += inputIndices[m].size();
		numOutputs += outputIndices[m].size();
	}

	// width, height and length of output region
	int l = fMax - fMin + 1;
	int h = iMax - iMin + 1;
	int w = jMax - jMin + 1;

	if(w < 1)
		throw Exception("There needs to be at least one active pixel in the output mask.");

	if(numOutputs != w * h * l)
		throw Exception("Unsupported output mask.");

	if(preconditioner) {
		if(numInputs != preconditioner->dimIn() || numOutputs != preconditioner->dimOut())
			throw Exception("Preconditioner and masks are incompatible.");
		if(preconditioner->dimInPre() != model.dimIn() || preconditioner->dimOutPre() != model.dimOut())
			throw Exception("Model and preconditioner are incompatible.");
	} else {
		if(numInputs != model.dimIn() || numOutputs != model.dimOut())
			throw Exception("Model and masks are incompatible.");
	}

	for(int f = 0; f + inputMask.size() <= video.size(); f += l)
		for(int i = 0; i + inputMask[0].rows() <= video[0].rows(); i += h)
			for(int j = 0; j + inputMask[0].cols() <= video[0].cols(); j += w) {
				VectorXd input(numInputs);

				// extract causal neighborhood
				for(int m = 0, offset = 0; m < inputMask.size(); ++m) {
					input.segment(offset, inputIndices[m].size()) = extractFromImage(
						video[f + m].block(i, j, inputMask[m].rows(), inputMask[m].cols()), inputIndices[m]);
					offset += inputIndices[m].size();
				}

				// sample output
				VectorXd output;

				if(preconditioner) {
					input = preconditioner->operator()(input);
					output = preconditioner->inverse(input, model.sample(input)).second;
				} else {
					output = model.sample(input);
				}

				// replace pixels in video by model's output
				for(int m = 0, offset = 0; m < inputMask.size(); ++m) {
					for(int k = 0; k < outputIndices[m].size(); ++k)
						video[f + m](i + outputIndices[m][k].first, j + outputIndices[m][k].second) = output[offset + k];
					offset += outputIndices[m].size();
				}
			}

	return video;
}



inline double computeEnergy(
	const ArrayXXd& img,
	const ConditionalDistribution& model,
	const ArrayXXb& inputMask,
	const ArrayXXb& outputMask,
	const Tuples& inputIndices,
	const Tuples& outputIndices,
	const Preconditioner* preconditioner,
	int i,
	int j,
	const Tuples& offsets)
{
	double energy = 0.;

	for(Tuples::const_iterator offset = offsets.begin(); offset != offsets.end(); ++offset) {
		if(i + offset->first < 0 || j + offset->second < 0 || i + offset->first + inputMask.rows() >= img.rows() || j + offset->second + inputMask.cols() >= img.cols()) {
			std::cout << i << ", " << j << ", " << offset->first << ", " << offset->second << std::endl;
		}

		// extract inputs and outputs from image
		ArrayXXd patch = img.block(
			i + offset->first,
			j + offset->second,
			inputMask.rows(),
			inputMask.cols());
		VectorXd input = extractFromImage(patch, inputIndices);
		VectorXd output = extractFromImage(patch, outputIndices);

		// update energy
		if(preconditioner) {
			pair<ArrayXXd, ArrayXXd> data = preconditioner->operator()(input, output);
			energy -= model.logLikelihood(data.first, data.second)[0] + preconditioner->logJacobian(input, output)[0];
		} else {
			energy -= model.logLikelihood(input, output)[0];
		}
	}

	return energy;
}



ArrayXXd CMT::fillInImage(
	ArrayXXd img,
	const ConditionalDistribution& model,
	const ArrayXXb& inputMask,
	const ArrayXXb& outputMask,
	const ArrayXXb& fillInMask,
	const Preconditioner* preconditioner,
	int numIterations,
	int numSteps)
{
	if(fillInMask.rows() != img.rows() || fillInMask.cols() != img.cols())
		throw Exception("Image and mask size incompatible.");

	Tuples fillInIndices = maskToIndices(fillInMask);
	pair<Tuples, Tuples> inOutIndices = masksToIndices(inputMask, outputMask);
	Tuples& inputIndices = inOutIndices.first;
	Tuples& outputIndices = inOutIndices.second;
	Tuples offsets;

	if(outputIndices.size() != 1)
		throw Exception("Only one-pixel output masks are currently supported.");

	// TODO: remove pixels from fillInIndices which cannot be predicted

	// compute offsets
	offsets.push_back(make_pair(-outputIndices[0].first, -outputIndices[0].second));
	for(int i = 0; i < inputIndices.size(); ++i)
		offsets.push_back(make_pair(-inputIndices[i].first, -inputIndices[i].second));

	for(int i = 0; i < numIterations; ++i)
		for(Tuples::iterator iter = fillInIndices.begin(); iter != fillInIndices.end(); ++iter) {
			// sample from cauchy distribution
			ArrayXd noise = sampleNormal(numSteps) / 4.;
			ArrayXd uni = ArrayXd::Random(numSteps).abs();

			double valueOld = img(iter->first, iter->second);
			double energyOld = computeEnergy(
				img, 
				model, 
				inputMask, outputMask,
				inputIndices, outputIndices,
				preconditioner,
				iter->first, iter->second,
				offsets);

			int counter = 0;

			for(int j = 0; j < numSteps; ++j) {
				// proposal sample
				img(iter->first, iter->second) = valueOld + noise[j];

				double energyNew = computeEnergy(
					img, 
					model, 
					inputMask, outputMask,
					inputIndices, outputIndices,
					preconditioner,
					iter->first, iter->second,
					offsets);

				if(uni[j] >= exp(energyOld - energyNew)) {
					// reject proposed step
					img(iter->first, iter->second) = valueOld;
				} else {
					// accept proposed step
					valueOld = img(iter->first, iter->second);
					energyOld = energyNew;
					++counter;
				}
			}

			std::cout << "acceptance rate: " << static_cast<double>(counter) / numSteps << std::endl;
		}

	return img;
}



struct BFGSInstance {
	const Tuples* positions;
	const ConditionalDistribution* model;
	ArrayXXd* img;
	const Tuples* inputIndices;
	const Tuples* outputIndices;
	const Tuples* block;
	const ArrayXXb* inputMask;
	const Preconditioner* preconditioner;
};



lbfgsfloatval_t fillInImageMAPGradient(
	void* instance,
	const lbfgsfloatval_t* x,
	lbfgsfloatval_t* g,
	const int, 
	const lbfgsfloatval_t step)
{
	const Tuples& positions = *static_cast<BFGSInstance*>(instance)->positions;
	const ConditionalDistribution& model = *static_cast<BFGSInstance*>(instance)->model;
	const Tuples& inputIndices = *static_cast<BFGSInstance*>(instance)->inputIndices;
	const Tuples& outputIndices = *static_cast<BFGSInstance*>(instance)->outputIndices;
	const Tuples& block = *static_cast<BFGSInstance*>(instance)->block;
	ArrayXXd& img = *static_cast<BFGSInstance*>(instance)->img;
	const ArrayXXb& inputMask = *static_cast<BFGSInstance*>(instance)->inputMask;
	const Preconditioner* preconditioner = static_cast<BFGSInstance*>(instance)->preconditioner;

	// extract relevant inputs and outputs from image
	ArrayXXd inputs(inputIndices.size(), positions.size());
	ArrayXXd outputs(outputIndices.size(), positions.size());

	// load current state of pixels into image
	for(int i = 0; i < block.size(); ++i)
		img(block[i].first, block[i].second) = x[i];

	for(int i = 0; i < positions.size(); ++i) {
		ArrayXXd patch = img.block(
			positions[i].first,
			positions[i].second,
			inputMask.rows(),
			inputMask.cols());
		inputs.col(i) = extractFromImage(patch, inputIndices);
		outputs.col(i) = extractFromImage(patch, outputIndices);
	}

	// compute gradients
	pair<pair<ArrayXXd, ArrayXXd>, Array<double, 1, Dynamic> > results;
	
	if(preconditioner) {
		pair<ArrayXXd, ArrayXXd> data = preconditioner->operator()(inputs, outputs);
		results = model.computeDataGradient(data.first, data.second);

		// adjust gradient and likelihood to take transformation into account
		results.first = preconditioner->adjustGradient(results.first.first, results.first.second);
		results.second += preconditioner->logJacobian(inputs, outputs);
	} else {
		results = model.computeDataGradient(inputs, outputs);
	}

	ArrayXXd& inputGradient = results.first.first;
	ArrayXXd& outputGradient = results.first.second;
	Array<double, 1, Dynamic>& logLikelihood = results.second;

	// combine gradients
	ArrayXXd gradient = ArrayXXd::Zero(img.rows(), img.cols());

	for(int i = 0; i < positions.size(); ++i) {
		Block<ArrayXXd> patch = gradient.block(
			positions[i].first,
			positions[i].second,
			inputMask.rows(),
			inputMask.cols());

		for(int j = 0; j < inputIndices.size(); ++j)
			patch(inputIndices[j].first, inputIndices[j].second) += inputGradient(j, i);

		for(int j = 0; j < outputIndices.size(); ++j)
			patch(outputIndices[j].first, outputIndices[j].second) += outputGradient(j, i);
	}

	// store relevant part of gradient
	if(g) {
		for(int i = 0; i < block.size(); ++i)
			g[i] = -gradient(block[i].first, block[i].second);
	}

	return -logLikelihood.sum();
}



ArrayXXd CMT::fillInImageMAP(
	ArrayXXd img,
	const ConditionalDistribution& model,
	const ArrayXXb& inputMask,
	const ArrayXXb& outputMask,
	const ArrayXXb& fillInMask,
	const Preconditioner* preconditioner,
	int numIterations,
	int patchSize)
{
	if(fillInMask.rows() != img.rows() || fillInMask.cols() != img.cols())
		throw Exception("Image and mask size incompatible.");

	pair<Tuples, Tuples> inOutIndices = masksToIndices(inputMask, outputMask);
	Tuples& inputIndices = inOutIndices.first;
	Tuples& outputIndices = inOutIndices.second;

	// TODO: allow for more general output masks
	if(outputIndices.size() != 1)
		throw Exception("Only one-pixel output masks are currently supported.");

	// TODO: remove pixels from fillInIndices which cannot be predicted (or extend image)

	// divide unobserved pixels into blocks
	vector<Tuples> blocks;

	// TODO: add missing blocks
	for(int i = 0; i < img.rows() - patchSize; i += patchSize) {
		for(int j = 0; j < img.cols() - patchSize; j += patchSize) {
			Tuples indices = maskToIndices(fillInMask.block(i, j, patchSize, patchSize));

			if(indices.size()) {
				for(Tuples::iterator it = indices.begin(); it != indices.end(); ++it) {
					it->first += i;
					it->second += j;
				}
				blocks.push_back(indices);
			}
		}
	}

	Tuples fillInIndices = maskToIndices(fillInMask);

	// precompute relative positions of neighborhoods which depend on a pixel
	Tuples offsets;
	offsets.push_back(make_pair(-outputIndices[0].first, -outputIndices[0].second));
	for(int i = 0; i < inputIndices.size(); ++i)
		offsets.push_back(make_pair(-inputIndices[i].first, -inputIndices[i].second));

	for(int i = 0; i < numIterations; ++i)
		// alternately optimize each block of pixels
		for(int j = 0; j < blocks.size(); ++j) {
			Tuples& block = blocks[j];

			// compute relevant neighborhood positions
			set<Tuple> uniquePositions;
			for(int k = 0; k < block.size(); ++k)
				for(int l = 0; l < offsets.size(); ++l)
					uniquePositions.insert(make_pair(
						block[k].first + offsets[l].first,
						block[k].second + offsets[l].second));

			// turn set into vector
			Tuples positions(uniquePositions.begin(), uniquePositions.end());

			// copy pixels into array
			lbfgsfloatval_t* x = lbfgs_malloc(block.size());

			for(int k = 0; k < block.size(); ++k)
				x[k] = img(block[k].first, block[k].second);

			// summarize variables needed to compute gradient
			BFGSInstance instance = {
				&positions, &model, &img, &inputIndices, &outputIndices, &block, &inputMask, preconditioner };

			// optimization hyperparameters
			lbfgs_parameter_t params;
			lbfgs_parameter_init(&params);
			params.max_iterations = 50;
			params.m = 6;
			params.epsilon = 1e-5;
			params.linesearch = LBFGS_LINESEARCH_MORETHUENTE;
			params.max_linesearch = 100;
			params.ftol = 1e-4;
			params.xtol = 1e-32;

			// start optimization
			lbfgs(block.size(), x, 0, &fillInImageMAPGradient, 0, &instance, &params);

			// copy pixels back
			for(int k = 0; k < block.size(); ++k)
				img(block[k].first, block[k].second) = x[k];

			lbfgs_free(x);
		}

	return img;
}



ArrayXXd CMT::extractWindows(const ArrayXXd& timeSeries, int windowLength) {
	ArrayXXd windows(timeSeries.rows() * windowLength, timeSeries.cols() - windowLength + 1);

	#ifndef EIGEN_DEFAULT_TO_ROW_MAJOR
	const double* dataFrom = timeSeries.data();
	double* dataTo = windows.data();
	#endif

	#pragma omp parallel for
	for(int t = 0; t < windows.cols(); ++t) {
		// read entries from time series in column-major order
		#ifdef EIGEN_DEFAULT_TO_ROW_MAJOR
		for(int j = 0, k = 0; j < windowLength; ++j)
			for(int i = 0; i < timeSeries.rows(); ++i, ++k)
				windows(k, t) = timeSeries(i, t + j);
		#else
		memcpy(
			dataTo + t * windows.rows(),
			dataFrom + t * timeSeries.rows(),
			sizeof(double) * windows.rows());
		#endif
	}

	return windows;
}



ArrayXXd CMT::sampleSpikeTrain(
	const ArrayXXd& stimuli,
	const ConditionalDistribution& model,
	int spikeHistory,
	const Preconditioner* preconditioner)
{
	if(spikeHistory <= 0)
		return model.sample(stimuli);

	// container for sampled spike train
	ArrayXXd spikeTrain = ArrayXXd::Zero(model.dimOut(), stimuli.cols());

	if(preconditioner) {
		// container for input to the model and spike history
		ArrayXd input(stimuli.rows() + preconditioner->dimInPre());
		ArrayXd spikes(spikeHistory * model.dimOut());

		for(int t = spikeHistory; t < stimuli.cols(); ++t) {
			// extract spike history
			for(int k = 0, i = t - spikeHistory; i < t; ++i)
				for(int j = 0; j < model.dimOut(); ++j, ++k)
					spikes[k] = spikeTrain(j, i);

			// transform spike history
			ArrayXd spikesPrec = preconditioner->operator()(spikes);

			// copy stimulus and transformed spike history into input
			for(int i = 0; i < stimuli.rows(); ++i)
				input[i] = stimuli(i, t);
			for(int i = 0; i < spikesPrec.rows(); ++i)
				input[stimuli.rows() + i] = spikesPrec[i];

			spikeTrain.col(t) = model.sample(input);
		}
	} else {
		ArrayXd input(stimuli.rows() + spikeHistory * model.dimOut());

		for(int t = spikeHistory; t < stimuli.cols(); ++t) {
			// copy stimulus and spike history into input
			for(int i = 0; i < stimuli.rows(); ++i)
				input[i] = stimuli(i, t);
			for(int i = t - spikeHistory, k = stimuli.rows(); i < t; ++i)
				for(int j = 0; j < model.dimOut(); ++j, ++k)
					input[k] = spikeTrain(j, i);

			spikeTrain.col(t) = model.sample(input);
		}
	}

	return spikeTrain;
}
