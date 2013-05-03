#ifndef CMT_TOOLS_H
#define CMT_TOOLS_H

#include <vector>
#include "conditionaldistribution.h"
#include "preconditioner.h"
#include "Eigen/Core"

namespace CMT {
	using std::vector;
	using std::pair;

	using Eigen::VectorXd;
	using Eigen::ArrayXXd;
	using Eigen::Array;
	using Eigen::Dynamic;

	typedef Array<bool, Dynamic, Dynamic> ArrayXXb;
	typedef pair<int, int> Tuple;
	typedef vector<Tuple> Tuples;

	Tuples maskToIndices(const ArrayXXb& mask);
	pair<Tuples, Tuples> masksToIndices(
		const ArrayXXb& inputMask,
		const ArrayXXb& outputMask);

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
}

#endif
