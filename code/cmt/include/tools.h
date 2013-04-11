#ifndef TOOLS_H
#define TOOLS_H

#include <vector>
using std::vector;
using std::pair;

#include <utility>
using std::make_pair;

#include "conditionaldistribution.h"

#include "preconditioner.h"
using CMT::Preconditioner;

#include "Eigen/Core"
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
	ArrayXXd img,
	ArrayXXb inputMask,
	ArrayXXb outputMask,
	int numSamples);
pair<ArrayXXd, ArrayXXd> generateDataFromImage(
	vector<ArrayXXd> img,
	ArrayXXb inputMask,
	ArrayXXb outputMask,
	int numSamples);
pair<ArrayXXd, ArrayXXd> generateDataFromImage(
	vector<ArrayXXd> img,
	vector<ArrayXXb> inputMask,
	vector<ArrayXXb> outputMask,
	int numSamples);

pair<ArrayXXd, ArrayXXd> generateDataFromVideo(
	vector<ArrayXXd> video,
	vector<ArrayXXb> inputMask,
	vector<ArrayXXb> outputMask,
	int numSamples);

ArrayXXd sampleImage(
	ArrayXXd img,
	const ConditionalDistribution& model,
	ArrayXXb inputMask,
	ArrayXXb outputMask,
	const Preconditioner* preconditioner = 0);
vector<ArrayXXd> sampleImage(
	vector<ArrayXXd> img,
	const ConditionalDistribution& model,
	ArrayXXb inputMask,
	ArrayXXb outputMask,
	const Preconditioner* preconditioner = 0);
vector<ArrayXXd> sampleImage(
	vector<ArrayXXd> img,
	const ConditionalDistribution& model,
	vector<ArrayXXb> inputMask,
	vector<ArrayXXb> outputMask,
	const Preconditioner* preconditioner = 0);

vector<ArrayXXd> sampleVideo(
	vector<ArrayXXd> video,
	const ConditionalDistribution& model,
	vector<ArrayXXb> inputMask,
	vector<ArrayXXb> outputMask,
	const Preconditioner* preconditioner = 0);

ArrayXXd fillInImage(
	ArrayXXd img,
	const ConditionalDistribution& model,
	ArrayXXb inputMask,
	ArrayXXb outputMask,
	ArrayXXb fillInMask,
	const Preconditioner* preconditioner = 0,
	int numIterations = 10,
	int numSteps = 100);
ArrayXXd fillInImageMAP(
	ArrayXXd img,
	const ConditionalDistribution& model,
	ArrayXXb inputMask,
	ArrayXXb outputMask,
	ArrayXXb fillInMask,
	const Preconditioner* preconditioner = 0,
	int numIterations = 10,
	int patchSize = 20);

#endif
