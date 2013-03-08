#ifndef TOOLS_H
#define TOOLS_H

#include <vector>
using std::vector;
using std::pair;

#include <utility>
using std::make_pair;

#include "conditionaldistribution.h"

#include "preconditioner.h"
using MCM::Preconditioner;

#include "Eigen/Core"
using Eigen::VectorXd;
using Eigen::ArrayXXd;
using Eigen::Array;
using Eigen::Dynamic;

typedef Array<bool, Dynamic, Dynamic> ArrayXXb;
typedef vector<pair<int, int> > Tuples;

VectorXd extractFromImage(ArrayXXd img, Tuples indices);
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

#endif
