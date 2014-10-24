
#include "callbackinterface.h"
#include "stm.h"

bool TrainableCallback::operator()(int cbIter, const CMT::Trainable& obj) {
    // Turn struct to object
    MEX::Function constructor("cmt.STM");

    MEX::Data handle(1);

    const CMT::STM& stm = dynamic_cast<const CMT::STM&>(obj);

    handle[0] = MEX::ObjectHandle<const CMT::STM>::share(&stm);

    MEX::Data args = constructor(1, handle);

    args.resize(2, true);

    args[0] = cbIter;

    const MEX::Data& result = mFunction(1, args);

    return result(0);
}
