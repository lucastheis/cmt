#ifndef TRAINABLE_INTERFACE_H
#define TRAINABLE_INTERFACE_H

#include "trainable.h"

bool trainableParameters(CMT::Trainable::Parameters* params, std::string key, MEX::Input::Getter value);

bool trainableParse(CMT::Trainable* obj, std::string cmd, const MEX::Output& output, const MEX::Input& input);


#endif
