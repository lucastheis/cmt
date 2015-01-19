// Official mex interface
#include "mex.h"

// Unofficial C++ mex extension
#include "mex.hpp"

// The class we are going to wrap
#include "glm.h"
#include "conditionaldistributioninterface.h"
#include "trainableinterface.h"

CMT::GLM* mexCreate(const MEX::Input& input) {
    if(input.size() > 1)
        mexWarnMsgIdAndTxt("mexWrapper:ignoredArgurments", "Setting nonlinearity and distribution not supported yet.");

    return new CMT::GLM(input[0]);
}

bool mexParse(CMT::GLM* obj, std::string cmd, const MEX::Output& output, const MEX::Input& input) {
    // Parameter setter and getter
    if(cmd == "bias") {
        output[0] = obj->bias();
        return true;
    }

    if(cmd == "setBias") {
        obj->setBias(input[0]);
        return true;
    }

    if(cmd == "weights") {
        output[0] = obj->weights();
        return true;
    }

    if(cmd == "setWeights") {
        obj->setWeights(input[0]);
        return true;
    }

    // Superclasses
    if(conditionaldistributioninterface(obj, cmd, output, input))
        return true;

    if(trainableinterface(obj, cmd, output, input))
        return true;

    return false;
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    mexWrapper<CMT::GLM>(&mexCreate, &mexParse, nlhs, plhs, nrhs, prhs);
}


