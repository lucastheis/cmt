
#include "callbackinterface.h"

bool TrainableCallback::operator()(int cbIter, const CMT::Trainable& obj) {

    MEX::Data args(2);

    args[0] = cbIter;

    // Testing structs //ToDo: Find out why this does not work!
    const char *field_names[] = {"dimIn", "dimOut"};
    mxArray* test_struct = mxCreateStructMatrix(1, 1, 2, field_names);

    mxArray *dim_in;
    dim_in = mxCreateNumericMatrix(1, 1, mxUINT32_CLASS, mxREAL);
    *((int*) mxGetData(dim_in)) = obj.dimIn();
    mxSetFieldByNumber(test_struct, 0, 0, dim_in);

    mxArray *dim_out;
    dim_out = mxCreateNumericMatrix(1, 1, mxUINT32_CLASS, mxREAL);
    *((int*) mxGetData(dim_out)) = obj.dimOut();
    mxSetFieldByNumber(test_struct, 0, 1, dim_out);

    args[1] = (mxArray*) test_struct;

    mxDestroyArray(test_struct);

    const MEX::Data& result = mFunction(1, args);

    return result(0);
}
