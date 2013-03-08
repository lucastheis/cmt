#include <algorithm>
using std::max;
using std::min;

#include <cstdio>
using std::rand;

#include <cmath>
using std::exp;

#include "tools.h"
#include "utils.h"
#include "exception.h"

using Eigen::Block;
using Eigen::ArrayXd;

#include <iostream>

VectorXd extractFromImage(ArrayXXd img, Tuples indices) {
	VectorXd pixels(indices.size());

	for(int i = 0; i < indices.size(); ++i)
		pixels[i] = img(indices[i].first, indices[i].second);

	return pixels;
}



Tuples maskToIndices(const ArrayXXb& mask) {
	Tuples indices;

	for(int i = 0; i < mask.rows(); ++i)
		for(int j = 0; j < mask.cols(); ++j)
			if(mask(i, j))
				indices.push_back(make_pair(i, j));

	return indices;
}



pair<Tuples, Tuples> masksToIndices(const ArrayXXb& inputMask, const ArrayXXb& outputMask) {
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



pair<ArrayXXd, ArrayXXd> generateDataFromImage(
	ArrayXXd img,
	ArrayXXb inputMask,
	ArrayXXb outputMask,
	int numSamples)
{
	int w = img.cols() - inputMask.cols() + 1;
	int h = img.rows() - inputMask.rows() + 1;

	if(numSamples > w * h)
		throw Exception("Image not large enough for this many samples.");

	// precompute indices of active pixels in masks
	pair<Tuples, Tuples> inOutIndices = masksToIndices(inputMask, outputMask);
	Tuples& inputIndices = inOutIndices.first;
	Tuples& outputIndices = inOutIndices.second;

	// sample random image locations
	set<int> indices = randomSelect(numSamples, w * h);

	pair<ArrayXXd, ArrayXXd> data = make_pair(
		ArrayXXd(inputIndices.size(), numSamples),
		ArrayXXd(outputIndices.size(), numSamples));

	int k = 0;

	for(set<int>::iterator iter = indices.begin(); iter != indices.end(); ++iter, ++k) {
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



pair<ArrayXXd, ArrayXXd> generateDataFromImage(
	vector<ArrayXXd> img,
	ArrayXXb inputMask,
	ArrayXXb outputMask,
	int numSamples)
{
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

	int numInputs = inputIndices.size();
	int numOutputs = outputIndices.size();

	pair<ArrayXXd, ArrayXXd> data = make_pair(
		ArrayXXd(numChannels * numInputs, numSamples),
		ArrayXXd(numChannels * numOutputs, numSamples));

	int k = 0;

	for(set<int>::iterator iter = indices.begin(); iter != indices.end(); ++iter, ++k) {
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



pair<ArrayXXd, ArrayXXd> generateDataFromImage(
	vector<ArrayXXd> img,
	vector<ArrayXXb> inputMask,
	vector<ArrayXXb> outputMask,
	int numSamples)
{
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

	pair<ArrayXXd, ArrayXXd> data = make_pair(
		ArrayXXd(numInputs, numSamples),
		ArrayXXd(numOutputs, numSamples));

	int k = 0;

	for(set<int>::iterator iter = indices.begin(); iter != indices.end(); ++iter, ++k) {
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



pair<ArrayXXd, ArrayXXd> generateDataFromVideo(
	vector<ArrayXXd> video,
	vector<ArrayXXb> inputMask,
	vector<ArrayXXb> outputMask,
	int numSamples)
{
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

	pair<ArrayXXd, ArrayXXd> data = make_pair(
		ArrayXXd(numInputs, numSamples),
		ArrayXXd(numOutputs, numSamples));

	int k = 0;

	for(set<int>::iterator iter = indices.begin(); iter != indices.end(); ++iter, ++k) {
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



ArrayXXd sampleImage(
	ArrayXXd img,
	const ConditionalDistribution& model,
	ArrayXXb inputMask,
	ArrayXXb outputMask,
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



vector<ArrayXXd> sampleImage(
	vector<ArrayXXd> img,
	const ConditionalDistribution& model,
	ArrayXXb inputMask,
	ArrayXXb outputMask,
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



vector<ArrayXXd> sampleImage(
	vector<ArrayXXd> img,
	const ConditionalDistribution& model,
	vector<ArrayXXb> inputMask,
	vector<ArrayXXb> outputMask,
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
		if(img[m].cols() != img[0].cols() || img[m].rows() != img[0].cols())
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



vector<ArrayXXd> sampleVideo(
	vector<ArrayXXd> video,
	const ConditionalDistribution& model,
	vector<ArrayXXb> inputMask,
	vector<ArrayXXb> outputMask,
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



ArrayXXd fillInImage(
	ArrayXXd img,
	const ConditionalDistribution& model,
	ArrayXXb inputMask,
	ArrayXXb outputMask,
	ArrayXXb fillInMask,
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

	// TODO: remove pixels from fillInIndices which cannot be predicted

	if(outputIndices.size() != 1)
		throw Exception("Only one-pixel output masks are currently supported.");

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
