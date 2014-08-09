// Official mex interface
#include "mex.h"

#include "stm.h"
using CMT::STM;

void demo_helper() {
    CMT::Trainable::Parameters* train_param = new CMT::STM::Parameters();

    CMT::STM::Parameters* stm_param = dynamic_cast<CMT::STM::Parameters*>(train_param);

    if(stm_param == 0){
        mexWarnMsgIdAndTxt("mex:null_pointer", "Null pointer after dynamic_cast in helper.\n");
    }
}

