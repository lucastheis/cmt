#ifndef TOOLS_H
#define TOOLS_H

#include <vector>
using std::vector;
using std::pair;

#include <utility>
using std::make_pair;

#include "conditionaldistribution.h"
#include "identitytransform.h"
using MCM::IdentityTransform;

#include "transform.h"
using MCM::Transform;

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

ArrayXXd sampleImage(
	ArrayXXd img,
	const ConditionalDistribution& model,
	ArrayXXb inputMask,
	ArrayXXb outputMask,
	const Transform& preconditioner = IdentityTransform());
vector<ArrayXXd> sampleImage(
	vector<ArrayXXd> img,
	const ConditionalDistribution& model,
	ArrayXXb inputMask,
	ArrayXXb outputMask,
	const Transform& preconditioner = IdentityTransform());

#endif
