// Official mex interface
#include "mex.h"

// The class we are going to wrap
#include "stm.h"
using CMT::STM;

#include "Eigen/Core"
using Eigen::MatrixXd;
using Eigen::Map;
using Eigen::ColMajor;

#include "cmt_demo_helper.h"

// Give matlab something to call
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    STM model(10);

    MatrixXd stimulus = MatrixXd::Random(10, 1000);
    MatrixXd spikes   = MatrixXd::Random(1, 1000);

    try {
        model.train(stimulus, spikes);
    } catch (std::exception& e) {
        mexWarnMsgIdAndTxt("mex:exceptionCaught", "Exception in training: \n\t%s", e.what());
    }

    CMT::Trainable::Parameters* train_param = new CMT::STM::Parameters();

    CMT::STM::Parameters* stm_param = dynamic_cast<CMT::STM::Parameters*>(train_param);

    if(stm_param == 0){
        mexWarnMsgIdAndTxt("mex:null_pointer", "Null pointer after dynamic_cast.\n");
    }

    demo_helper();

    if(nlhs > 0){
        MatrixXd w = model.weights();
        (*plhs) = mxCreateDoubleMatrix(w.rows(), w.cols(), mxREAL);
        Map<MatrixXd,ColMajor> data_wrapper(mxGetPr(*plhs), w.rows(), w.cols());
        data_wrapper = w;
    }
}


