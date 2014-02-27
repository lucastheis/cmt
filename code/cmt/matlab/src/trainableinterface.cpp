#include "trainable.h"

#include "mex.h"
#include "mex.hpp"


bool trainableinterface(CMT::Trainable* obj, std::string cmd, MEXOutput output, MEXInput input) {

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
