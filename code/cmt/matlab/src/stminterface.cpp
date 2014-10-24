// Official mex interface
#include "mex.h"

// Unofficial C++ mex extension
#include "mex.hpp"

// The class we are going to wrap
#include "stm.h"
#include "conditionaldistributioninterface.h"
#include "trainableinterface.h"
#include "callbackinterface.h"

bool stmparameters(CMT::STM::Parameters* params, std::string key, MEX::Input::Getter value) {
    if(key == "trainSharpness") {
        params->trainSharpness = value;
        return true;
    }

    if(key == "trainBiases") {
        params->trainBiases = value;
        return true;
    }

    if(key == "trainWeights") {
        params->trainWeights = value;
        return true;
    }

    if(key == "trainFeatures") {
        params->trainFeatures = value;
        return true;
    }

    if(key == "trainPredictors") {
        params->trainPredictors = value;
        return true;
    }

    if(key == "trainLinearPredictor") {
        params->trainLinearPredictor = value;
        return true;
    }


    // if(key == "regularizeBiases") {
    //     params->regularizeBiases = value;
    //     return true;
    // }

    // if(key == "regularizeWeights") {
    //     params->regularizeWeights = value;
    //     return true;
    // }

    // if(key == "regularizeFeatures") {
    //     params->regularizeFeatures = value;
    //     return true;
    // }

    // if(key == "regularizePredictors") {
    //     params->regularizePredictors = value;
    //     return true;
    // }

    // if(key == "regularizeLinearPredictor") {
    //     params->regularizeLinearPredictor = value;
    //     return true;
    // }

    return trainableparameters(params, key, value);
}

CMT::STM* mexCreate(const MEX::Input& input) {
    if(input.size() > 4)
        mexWarnMsgIdAndTxt("mexWrapper:ignoredArgurments", "Setting nonlinearity and distribution not supported yet.");

    if(input.has(3) && input[3].isType(MEX::Type::IntScalar)) {
        return new CMT::STM(input[0], input[1], input[2], input[3]);
    }

    if(input.has(2)) {
        return new CMT::STM(input[0], input[1], input[2]);
    }

    if(input.has(1)) {
        return new CMT::STM(input[0], input[1]);
    }

    return new CMT::STM(input[0]);
}

bool mexParse(CMT::STM* obj, std::string cmd, const MEX::Output& output, const MEX::Input& input) {

    if(cmd == "callback") {
        TrainableCallback callback(input[0]);
        if(callback(10, *obj)) {
            std::cout << "Success!" << std::endl;
        } else {
            std::cout << "Fail!" << std::endl;
        }
        return true;
    }

    // Parameter setter and getter
    if(cmd == "sharpness") {
        output[0] = obj->sharpness();
        return true;
    }

    if(cmd == "setSharpness") {
        obj->setSharpness(input[0]);
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


    if(cmd == "features") {
        output[0] = obj->features();
        return true;
    }

    if(cmd == "setFeatures") {
        obj->setFeatures(input[0]);
        return true;
    }


    if(cmd == "predictors") {
        output[0] = obj->predictors();
        return true;
    }

    if(cmd == "setPredictors") {
        obj->setPredictors(input[0]);
        return true;
    }


    // Constant params
    if(cmd == "dimInLinear") {
        output[0] = obj->dimInLinear();
        return true;
    }

    if(cmd == "dimInNonlinear") {
        output[0] = obj->dimInNonlinear();
        return true;
    }

    if(cmd == "numComponents") {
        output[0] = obj->numComponents();
        return true;
    }

    if(cmd == "numFeatures") {
        output[0] = obj->numFeatures();
        return true;
    }


    // Methods
    if(cmd == "train") {
        bool converged;
        CMT::STM::Parameters params;

        // Check if user supplied a validation set
        if(input.has(3) && input[2].isType(MEX::Type::FloatMatrix) && input[3].isType(MEX::Type::FloatMatrix)) {

            // Check if there are extra parameters
            if(input.has(4)) {
                params = input.toStruct<CMT::STM::Parameters>(4, &stmparameters);
            }

            converged = obj->train(input[0], input[1], input[2], input[3], params);
        } else {

            // Check if there are extra parameters
            if(input.has(2)) {
                params = input.toStruct<CMT::STM::Parameters>(2, &stmparameters);
            }

            converged = obj->train(input[0], input[1], params);
        }

        if(output.has(0)) {
            output[0] = converged;
        }
        return true;
    }

    if(cmd == "response") {
        if(input.has(1)) {
            output[0] = obj->response(input[0], input[1]);
            return true;
        }

        output[0] = obj->response(input[0]);
        return true;
    }

    if(cmd == "nonlinearResponses") {
        output[0] = obj->nonlinearResponses(input[0]);
        return true;
    }

    if(cmd == "linearResponse") {
        output[0] = obj->linearResponse(input[0]);
        return true;
    }

    // Superclasses
    if(trainableinterface(obj, cmd, output, input))
        return true;

    if(conditionaldistributioninterface(obj, cmd, output, input))
        return true;

    // Got here, so command not recognized
    return false;
}

// Give matlab something to call
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    mexWrapper<CMT::STM>(&mexCreate, &mexParse, nlhs, plhs, nrhs, prhs);
}


