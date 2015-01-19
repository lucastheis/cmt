// Official mex interface
#include "mex.h"

// Unofficial C++ mex extension
#include "mex.hpp"

// The class we are going to wrap
#include "mcgsm.h"
#include "trainableinterface.h"

#include "callbackinterface.h"

bool mcgsmParameters(CMT::MCGSM::Parameters* params, std::string key, MEX::Input::Getter value) {

    if(key == "trainPriors") {
        params->trainPriors = value;
        return true;
    }

    if(key == "trainScales") {
        params->trainScales = value;
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

    if(key == "trainCholeskyFactors") {
        params->trainCholeskyFactors = value;
        return true;
    }

    if(key == "trainPredictors") {
        params->trainPredictors = value;
        return true;
    }

    if(key == "trainLinearFeatures") {
        params->trainLinearFeatures = value;
        return true;
    }

    if(key == "trainMeans") {
        params->trainMeans = value;
        return true;
    }

    if(key == "callback") {
        if(params->callback != NULL) {
            delete params->callback;
        }

        params->callback = new TrainableCallback<CMT::MCGSM>(MEX::Function("cmt.MCGSM"), value);
        return true;
    }

    return trainableParameters(params, key, value);
}

CMT::MCGSM* mcgsmCreate(const MEX::Input& input) {
    if(input.has(4)) {
        return new CMT::MCGSM(input[0], input[1], input[2], input[3], input[4]);
    }

    if(input.has(3)) {
        return new CMT::MCGSM(input[0], input[1], input[2], input[3]);
    }

    if(input.has(2)) {
        return new CMT::MCGSM(input[0], input[1], input[2]);
    }

    if(input.has(1)) {
        return new CMT::MCGSM(input[0], input[1]);
    }

    return new CMT::MCGSM(input[0]);
}

bool mcgsmParse(CMT::MCGSM* obj, std::string cmd, const MEX::Output& output, const MEX::Input& input) {

    // Constant parameters
    if(cmd == "numComponents") {
        output[0] = obj->numComponents();
        return true;
    }

    if(cmd == "numScales") {
        output[0] = obj->numScales();
        return true;
    }

    if(cmd == "numFeatures") {
        output[0] = obj->numFeatures();
        return true;
    }


    // Parameter setter and getter
    if(cmd == "priors") {
        output[0] = obj->priors();
        return true;
    }

    if(cmd == "setPriors") {
        obj->setPriors(input[0]);
        return true;
    }


    if(cmd == "scales") {
        output[0] = obj->scales();
        return true;
    }

    if(cmd == "setScales") {
        obj->setScales(input[0]);
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


    if(cmd == "choleskyFactors") {
        output[0] = obj->choleskyFactors();
        return true;
    }

    if(cmd == "setCholeskyFactors") {
        obj->setCholeskyFactors(input[0]);
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


    if(cmd == "linearFeatures") {
        output[0] = obj->linearFeatures();
        return true;
    }

    if(cmd == "setLinearFeatures") {
        obj->setLinearFeatures(input[0]);
        return true;
    }


    if(cmd == "means") {
        output[0] = obj->means();
        return true;
    }

    if(cmd == "setMeans") {
        obj->setMeans(input[0]);
        return true;
    }


     // Methods
    if(cmd == "train") {
        bool converged;
        CMT::MCGSM::Parameters params;

        // Check if user supplied a validation set
        if(input.has(3) && input[2].isType(MEX::Type::FloatMatrix) && input[3].isType(MEX::Type::FloatMatrix)) {

            // Check if there are extra parameters
            if(input.has(4)) {
                params = input.toStruct<CMT::MCGSM::Parameters>(4, &mcgsmParameters);
            }

            converged = obj->train(input[0], input[1], input[2], input[3], params);
        } else {

            // Check if there are extra parameters
            if(input.has(2)) {
                params = input.toStruct<CMT::MCGSM::Parameters>(2, &mcgsmParameters);
            }

            converged = obj->train(input[0], input[1], params);
        }

        if(output.has(0)) {
            output[0] = converged;
        }
        return true;
    }

    if(cmd == "sample") {
        if(input.has(1)) {
            output[0] = obj->sample(input[0], input[1]);
            return true;
        }

        output[0] = obj->sample(input[0]);
        return true;
    }

    if(cmd == "posterior") {
        output[0] = obj->posterior(input[0], input[1]);
        return true;
    }

    if(cmd == "prior") {
        output[0] = obj->prior(input[0]);
        return true;
    }

    if(cmd == "samplePosterior") {
        output[0] = (Eigen::ArrayXXi) obj->samplePosterior(input[0], input[1]);
        return true;
    }

    if(cmd == "samplePrior") {
        output[0] = (Eigen::ArrayXXi) obj->samplePrior(input[0]);
        return true;
    }

    // Superclasses
    return trainableParse(obj, cmd, output, input);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    mexWrapper<CMT::MCGSM>(&mcgsmCreate, &mcgsmParse, nlhs, plhs, nrhs, prhs);
}


