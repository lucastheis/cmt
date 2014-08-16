#include "trainable.h"

#include "mex.h"
#include "mex.hpp"

#include "callbackinterface.h"

bool trainableparameters(CMT::Trainable::Parameters* params, std::string key, MEX::Input::Getter value) {
    if(key == "verbosity") {
        params->verbosity = value;
        return true;
    }

    if(key == "maxIter") {
        params->maxIter = value;
        return true;
    }

    if(key == "threshold") {
        params->threshold = value;
        return true;
    }

    if(key == "numGrad") {
        params->numGrad = value;
        return true;
    }

    if(key == "batchSize") {
        params->batchSize = value;
        return true;
    }

    if(key == "callback") {
        if(params->callback != NULL) {
            delete params->callback;
        }
        
        params->callback = new TrainableCallback(value);
        return true;
    }

    if(key == "cbIter") {
        params->cbIter = value;
        return true;
    }

    if(key == "valIter") {
        params->valIter = value;
        return true;
    }

    if(key == "valLookAhead") {
        params->valLookAhead = value;
        return true;
    }

    return false;
}


bool trainableinterface(CMT::Trainable* obj, std::string cmd, const MEX::Output& output, const MEX::Input& input) {

    // Methods
    if(cmd == "initialize") {
        obj->initialize(input[0], input[1]);
        return true;
    }

    if(cmd == "train") {
        if(input.has(4)){
            mexWarnMsgIdAndTxt("mexWrapper:ignoredArgurments", "Setting parameters is not supported yet.");
        }

        if(input.has(3)){
            output[0] = obj->train(input[0], input[1], input[2], input[3]);
            return true;
        }

        if(input.has(2)){
            mexWarnMsgIdAndTxt("mexWrapper:ignoredArgurments", "Setting parameters is not supported yet.");
        }

        output[0] = obj->train(input[0], input[1], CMT::Trainable::Parameters()); // Empty parameters are needed to avoid function matching problems.
        return true;
    }

    if(cmd == "checkGradient") {
        if(input.has(3)){
            mexWarnMsgIdAndTxt("mexWrapper:ignoredArgurments", "Setting parameters is not supported yet.");
        }

        if(input.has(2)) {
            output[0] = obj->checkGradient(input[0], input[1], input[2]);
            return true;
        }

        output[0] = obj->checkGradient(input[0], input[1]);
        return true;
    }

    if(cmd == "checkPerformance") {
        if(input.has(3)){
            mexWarnMsgIdAndTxt("mexWrapper:ignoredArgurments", "Setting parameters is not supported yet.");
        }

        if(input.has(2)) {
            output[0] = obj->checkPerformance(input[0], input[1], input[2]);
            return true;
        }

        output[0] = obj->checkPerformance(input[0], input[1]);
        return true;
    }

    if(cmd == "fisherInformation") {
        if(input.has(2)){
            mexWarnMsgIdAndTxt("mexWrapper:ignoredArgurments", "Setting parameters is not supported yet.");
        }

        output[0] = obj->checkPerformance(input[0], input[1]);
        return true;
    }

    return false;
}
