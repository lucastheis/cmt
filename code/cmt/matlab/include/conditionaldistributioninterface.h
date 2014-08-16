#ifndef CONDITIONAL_DISTRIBUTION_INTERFACE_H
#define CONDITIONAL_DISTRIBUTION_INTERFACE_H

#include "mex.hpp"

#include "conditionaldistribution.h"

bool conditionaldistributioninterface(CMT::ConditionalDistribution* obj, std::string cmd, const MEX::Output& output, const MEX::Input& input);

#endif
