#ifndef TOOLS_H
#define TOOLS_H

#include <vector>
#include <utility>
#include "Eigen/Core"
#include "conditionaldistribution.h"

using std::vector;
using std::pair;
using std::make_pair;

typedef Array<bool, Dynamic, Dynamic> ArrayXXb;
typedef vector<pair<int, int> > Tuples;

VectorXd extractFromImage(ArrayXXd img, Tuples indices);

ArrayXXd sampleImage(
	ArrayXXd img,
	const ConditionalDistribution& model,
	ArrayXXb inputMask,
	ArrayXXb outputMask);

vector<ArrayXXd> sampleImage(
	vector<ArrayXXd> img,
	const ConditionalDistribution& model,
	ArrayXXb inputMask,
	ArrayXXb outputMask);

#endif
