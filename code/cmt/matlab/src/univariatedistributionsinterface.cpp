#include "univariatedistributionsinterface.h"

CMT::UnivariateDistribution* toDistribution(MEX::Input::Getter obj) {
    if(obj.isClass("cmt.Binomial")) {

        return new CMT::Binomial(obj.getObjectProperty("n"),
                              obj.getObjectProperty("p"));

    } else if(obj.isClass("cmt.Poisson")) {

        return new CMT::Poisson(obj.getObjectProperty("lambda"));

    } else if(obj.isClass("cmt.Bernoulli")) {

        return new CMT::Bernoulli(obj.getObjectProperty("prob"));

    } else {
        mexErrMsgIdAndTxt("univariateDistribution:toDistribution:invalidUnivariateDistribution",
                          "Unknown UnivariateDistribution '%s' supplied.", obj.getClass().c_str());
    }
    return NULL;
}

