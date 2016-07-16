#include "nonlinearitiesinterface.h"

CMT::Nonlinearity* toNonlinearity(MEX::Input::Getter obj) {
    if(obj.isClass("cmt.LogisticFunction")) {

        return new CMT::LogisticFunction(obj.getObjectProperty("epsilon"));

    } else if(obj.isClass("cmt.ExponentialFunction")) {

        return new CMT::ExponentialFunction(obj.getObjectProperty("epsilon"));

    } else if(obj.isClass("cmt.BlobNonlinearity")) {

        return new CMT::BlobNonlinearity(obj.getObjectProperty("numComponents"),
                                     obj.getObjectProperty("epsilon"));

    } else if(obj.isClass("cmt.TanhBlobNonlinearity")) {

        return new CMT::TanhBlobNonlinearity(obj.getObjectProperty("numComponents"),
                                         obj.getObjectProperty("epsilon"));

    } else if(obj.isClass("cmt.HistogramNonlinearity")) {

        return new CMT::HistogramNonlinearity(obj.getObjectProperty("inputs"),
                                          obj.getObjectProperty("outputs"),
                                          obj.getObjectProperty("numBins"),
                                          obj.getObjectProperty("epsilon"));

    } else {
        mexErrMsgIdAndTxt("nonlinearity:toNonlinearity:invalidNonlinearity",
                          "Unknown Nonlinearity '%s' supplied.", obj.getClass().c_str());
    }
    return NULL;
}
