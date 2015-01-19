// Official mex interface
#include "mex.h"

// Unofficial C++ mex extension
#include "mex.hpp"

// The class we are going to wrap
#include "glm.h"
#include "trainableinterface.h"

#include "callbackinterface.h"

bool glmParameters(CMT::GLM::Parameters* params, std::string key, MEX::Input::Getter value) {
    if(key == "callback") {
        if(params->callback != NULL) {
            delete params->callback;
        }

        params->callback = new TrainableCallback<CMT::GLM>(MEX::Function("cmt.GLM"), value);
        return true;
    }

    return trainableParameters(params, key, value);
}

CMT::GLM* glmCreate(const MEX::Input& input) {
    if(input.size() > 1)
        mexWarnMsgIdAndTxt("mexWrapper:ignoredArgurments", "Setting nonlinearity and distribution not supported yet.");

    return new CMT::GLM(input[0]);
}

bool glmParse(CMT::GLM* obj, std::string cmd, const MEX::Output& output, const MEX::Input& input) {
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
        output[0] = (Eigen::MatrixXd) obj->weights();
        return true;
    }

    if(cmd == "setWeights") {
        obj->setWeights(input[0]);
        return true;
    }

    // Methods
    if(cmd == "train") {
        bool converged;
        CMT::GLM::Parameters params;

        // Check if user supplied a validation set
        if(input.has(3) && input[2].isType(MEX::Type::FloatMatrix) && input[3].isType(MEX::Type::FloatMatrix)) {

            // Check if there are extra parameters
            if(input.has(4)) {
                params = input.toStruct<CMT::GLM::Parameters>(4, &glmParameters);
            }

            converged = obj->train(input[0], input[1], input[2], input[3], params);
        } else {

            // Check if there are extra parameters
            if(input.has(2)) {
                params = input.toStruct<CMT::GLM::Parameters>(2, &glmParameters);
            }

            converged = obj->train(input[0], input[1], params);
        }

        if(output.has(0)) {
            output[0] = converged;
        }
        return true;
    }


    // Superclasses
    return trainableParse(obj, cmd, output, input);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    mexWrapper<CMT::GLM>(&glmCreate, &glmParse, nlhs, plhs, nrhs, prhs);
}


