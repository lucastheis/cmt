#ifndef CMT_TOOLS_H
#define CMT_TOOLS_H

#include <vector>
#include "conditionaldistribution.h"
#include "mcgsm.h"
#include "preconditioner.h"
#include "Eigen/Core"

namespace Eigen {
	typedef Array<bool, Dynamic, Dynamic> ArrayXXb;
}

namespace CMT {
	using std::vector;
	using std::pair;

	using Eigen::ArrayXXb;
	using Eigen::ArrayXXi;
	using Eigen::VectorXd;
	using Eigen::ArrayXXd;
	using Eigen::Array;
	using Eigen::Dynamic;

	typedef pair<int, int> Tuple;
	typedef vector<Tuple> Tuples;

	Tuples maskToIndices(const ArrayXXb& mask);
	pair<Tuples, Tuples> masksToIndices(
		const ArrayXXb& inputMask,
		const ArrayXXb& outputMask);

	/**
	 * Extracts pixels from an image and returns them in a vector.
	 * 
	 * The order of the pixels in the vector corresponds to the order of the pixels
	 * in the given list of indices. Function odes not test for validity of indices.
	 * 
	 * @param img image from which to extract pixels
	 * @param indices list of pixel locations
	 */
	VectorXd extractFromImage(const ArrayXXd& img, const Tuples& indices);

	pair<ArrayXXd, ArrayXXd> generateDataFromImage(
		const ArrayXXd& img,
		const ArrayXXb& inputMask,
		const ArrayXXb& outputMask);
	pair<ArrayXXd, ArrayXXd> generateDataFromImage(
		const ArrayXXd& img,
		const ArrayXXb& inputMask,
		const ArrayXXb& outputMask,
		int numSamples);
	pair<ArrayXXd, ArrayXXd> generateDataFromImage(
		const vector<ArrayXXd>& img,
		const ArrayXXb& inputMask,
		const ArrayXXb& outputMask);
	pair<ArrayXXd, ArrayXXd> generateDataFromImage(
		const vector<ArrayXXd>& img,
		const ArrayXXb& inputMask,
		const ArrayXXb& outputMask,
		int numSamples);
	pair<ArrayXXd, ArrayXXd> generateDataFromImage(
		const vector<ArrayXXd>& img,
		const vector<ArrayXXb>& inputMask,
		const vector<ArrayXXb>& outputMask);
	pair<ArrayXXd, ArrayXXd> generateDataFromImage(
		const vector<ArrayXXd>& img,
		const vector<ArrayXXb>& inputMask,
		const vector<ArrayXXb>& outputMask,
		int numSamples);

	pair<ArrayXXd, ArrayXXd> generateDataFromVideo(
		const vector<ArrayXXd>& video,
		const vector<ArrayXXb>& inputMask,
		const vector<ArrayXXb>& outputMask);
	pair<ArrayXXd, ArrayXXd> generateDataFromVideo(
		const vector<ArrayXXd>& video,
		const vector<ArrayXXb>& inputMask,
		const vector<ArrayXXb>& outputMask,
		int numSamples);

	ArrayXXd sampleImage(
		ArrayXXd img,
		const ConditionalDistribution& model,
		const ArrayXXb& inputMask,
		const ArrayXXb& outputMask,
		const Preconditioner* preconditioner = 0);
	vector<ArrayXXd> sampleImage(
		vector<ArrayXXd> img,
		const ConditionalDistribution& model,
		const ArrayXXb& inputMask,
		const ArrayXXb& outputMask,
		const Preconditioner* preconditioner = 0);
	vector<ArrayXXd> sampleImage(
		vector<ArrayXXd> img,
		const ConditionalDistribution& model,
		const vector<ArrayXXb>& inputMask,
		const vector<ArrayXXb>& outputMask,
		const Preconditioner* preconditioner = 0);

	ArrayXXd sampleImageConditionally(
		ArrayXXd img,
		ArrayXXi labels,
		const MCGSM& model,
		const ArrayXXb& inputMask,
		const ArrayXXb& outputMask,
		const Preconditioner* preconditioner = 0,
		int numIter = 10,
		bool initialize = false);

	ArrayXXi sampleLabelsConditionally(
		ArrayXXd img,
		const MCGSM& model,
		const ArrayXXb& inputMask,
		const ArrayXXb& outputMask,
		const Preconditioner* preconditioner = 0);

	vector<ArrayXXd> sampleVideo(
		vector<ArrayXXd> video,
		const ConditionalDistribution& model,
		const vector<ArrayXXb>& inputMask,
		const vector<ArrayXXb>& outputMask,
		const Preconditioner* preconditioner = 0);

	ArrayXXd fillInImage(
		ArrayXXd img,
		const ConditionalDistribution& model,
		const ArrayXXb& inputMask,
		const ArrayXXb& outputMask,
		const ArrayXXb& fillInMask,
		const Preconditioner* preconditioner = 0,
		int numIterations = 10,
		int numSteps = 100);
	ArrayXXd fillInImageMAP(
		ArrayXXd img,
		const ConditionalDistribution& model,
		const ArrayXXb& inputMask,
		const ArrayXXb& outputMask,
		const ArrayXXb& fillInMask,
		const Preconditioner* preconditioner = 0,
		int numIterations = 10,
		int patchSize = 20);

	ArrayXXd extractWindows(const ArrayXXd& timeSeries, int windowLength);

	ArrayXXd sampleSpikeTrain(
		const ArrayXXd& stimuli,
		const ConditionalDistribution& model,
		int spikeHistory = 0,
		const Preconditioner* preconditioner = 0);
}

#endif
