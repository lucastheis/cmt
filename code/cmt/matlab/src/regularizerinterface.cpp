#include "regularizerinterface.h"

CMT::Regularizer::Norm toNorm(MEX::Input::Getter prop) {
    std::string s = prop;
    if(s == "L1") {
        return CMT::Regularizer::L1;
    } else if(s == "L2") {
        return CMT::Regularizer::L2;
    } else {
        mexErrMsgIdAndTxt("regularizer:toNorm:invalidNorm",
                          "Unknown norm '%s' supplied.", s.c_str());
    }

}

CMT::Regularizer toRegularizer(MEX::Input::Getter obj) {
    if(!obj.isClass("cmt.Regularizer")) {
        mexErrMsgIdAndTxt("regularizer:toRegularizer:unknownObject",
                          "Can not use objects of class '%s' as a regularizer.", obj.getClass().c_str());
    }

    return CMT::Regularizer((double) obj.getObjectProperty("strength"),
                            toNorm(obj.getObjectProperty("norm")));

}
