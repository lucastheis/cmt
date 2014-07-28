#include "conditionaldistribution.h"

#include "mex.h"
#include "mex.hpp"

bool conditionaldistributioninterface(CMT::ConditionalDistribution* obj, std::string cmd, MEXOutput output, MEXInput input) {

    // Parameter setter and getter
    if(cmd == "dimIn") {
        output[0] = obj->dimIn();
        return true;
    }

    if(cmd == "dimOut") {
        output[0] = obj->dimOut();
        return true;
    }

    // Methods
    if(cmd == "sample") {
        output[0] = obj->sample(input[0]);
        return true;
    }

    if(cmd == "predict") {
        output[0] = obj->predict(input[0]);
        return true;
    }

    if(cmd == "logLikelihood") {
        output[0] = obj->logLikelihood(input[0], input[1]);
        return true;
    }

    if(cmd == "evaluate") {
        if(input.has(2) && !input[2].isEmpty())
            mexWarnMsgTxt("Changing the preconditioner is currently not supported.");

        output[0] = obj->evaluate(input[0], input[1]);
        return true;
    }

    return false;
}
