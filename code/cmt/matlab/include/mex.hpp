/*
Copyright (c) 2012, Oliver Woodford
Copyright (c) 2014, Florian Franzen
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the distribution

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef __MEX_HPP__
#define __MEX_HPP__

#include "mex.h"

#include "MEX/ObjectHandle.h"
#include "MEX/Input.h"
#include "MEX/Output.h"

#include <stdint.h>
#include <string>
#include <cstring>
#include <iostream>
#include <sstream>
#include <exception>


// streambuffer that writes everything to mexPrintf. Useful to reroute cout to matlab.
class mexstreambuf : public std::streambuf {
protected:
    std::streamsize xsputn(const char *s, std::streamsize n) {
        mexPrintf("%.*s",n,s);
        return n;
    }

    int overflow(int c = EOF) {
        if (c != EOF) {
            mexPrintf("%.1s",&c);
        }
        return 1;
    }

};

/**
 * Mex wrapper function, that takes care of object creation, storage and deletion and passes everything else onto parser function.
 *
 * @param creator function that parses matlab data to call class constructor
 * @param parser function that parses matlab data and forwards the appropriate function
 */
template<class BaseClass> inline void mexWrapper(BaseClass* (*creator)(const MEX::Input&), bool (*parser)(BaseClass*, std::string, const MEX::Output&, const MEX::Input&), int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Redirect cout
    mexstreambuf mexoutbuf;
    std::streambuf* coutbuf = std::cout.rdbuf(&mexoutbuf);

    // Get the command string // ToDo: Maybe use mxArrayToString here, to drop 64 character limitation.
    char cmd[64];
    if (nrhs < 1 || mxGetString(prhs[0], cmd, sizeof(cmd)))
        mexErrMsgIdAndTxt("mexWrapper:invalidCommandstring", "First input should be a command string less than 64 characters long.");

    std::cout << "+++ DEBUG: Call to '" << cmd << "' with " << nrhs << " inputs and " << nlhs << " output value(s). +++" << std::endl;

    // New
    if (!strcmp("new", cmd)) {
        // Check parameters
        if (nlhs != 1)
            mexErrMsgIdAndTxt("mexWrapper:invalidOutput", "New: One output expected.");

        try {
            BaseClass* ptr = creator(MEX::Input(nrhs - 1, prhs + 1));
            // Return a handle to a new C++ instance
            plhs[0] = MEX::ObjectHandle<BaseClass>::wrap(ptr);
            return;
        } catch (std::exception& e) {
            mexErrMsgIdAndTxt("mexWrapper:constructor:exceptionCaught", "Exception in constructor: \n\t%s", e.what());
        }
    }

    // Check there is a second input, which should be the class instance handle
    if (nrhs < 2)
        mexErrMsgIdAndTxt("mexWrapper:missingObjectHandle", "Second input should be a class instance handle.");

    // Delete
    if (!strcmp("delete", cmd)) {
        // Destroy the C++ object
        MEX::ObjectHandle<BaseClass>::free(prhs[1]);
        // Warn if other commands were ignored
        if (nlhs != 0 || nrhs != 2)
            mexWarnMsgIdAndTxt("mexWrapper:ignoredArgurments", "Delete: Unexpected arguments ignored.");
        return;
    }

    // Get the class instance pointer from the second input
    BaseClass* instance = MEX::ObjectHandle<BaseClass>::unwrap(prhs[1]);

    try {
        if(!parser(instance, cmd, MEX::Output(nlhs, plhs), MEX::Input(nrhs - 2, prhs + 2)))
            mexErrMsgIdAndTxt("mexParse:unknownCommand", "Command '%s' not recognized!", cmd);
    } catch (std::exception& e) {
        mexErrMsgIdAndTxt("mexWrapper:methodAndPropertyParser:exceptionCaught", "Exception in method and property parser: \n\t%s", e.what());
    }

    // Reset cout
    std::cout.rdbuf(coutbuf);
}

#endif // __MEX_HPP__
