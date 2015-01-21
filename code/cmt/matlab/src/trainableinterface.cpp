// Official mex interface
#include "mex.h"

// Unofficial C++ mex extension
#include "mex.hpp"

// The class we are going to wrap
#include "trainable.h"
#include "conditionaldistributioninterface.h"

#include "callbackinterface.h"

bool trainableParameters(CMT::Trainable::Parameters* params, std::string key, MEX::Input::Getter value) {
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

bool trainableParse(CMT::Trainable* obj, std::string cmd, const MEX::Output& output, const MEX::Input& input) {

    // Methods
    if(cmd == "initialize") {
        obj->initialize(input[0], input[1]);
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

    // Superclass
    return conditionaldistributionParse(obj, cmd, output, input);
}
