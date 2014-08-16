#ifndef __MEX_FUNCTION_H__
#define __MEX_FUNCTION_H__

#include "Data.h"

#include <iostream>

#include "mex.h"

namespace MEX {
	class Function {
	public:
		Function(const mxArray* handle) : mHandle(handle), mClassID(mxGetClassID(handle)){
			if(mClassID != mxFUNCTION_CLASS && mClassID != mxCHAR_CLASS) {
		    	mexErrMsgIdAndTxt("MEX:Function:invalidFunctionHandle", "Supplied argument must be a function handle or string.");
			}
		}

		mxArray* exception = NULL;
		const Data operator()(int ret_count, Data args) {
			// Allocate return values
			Data result(ret_count);

			if(mClassID == mxFUNCTION_CLASS) {
				// Prepare input by adding function handle to front
				args.resize(args.size() + 1, true);

				args[0] = mHandle;

			    // Execute and check for exceptions
			    exception = mexCallMATLABWithTrap(result.size(), result, args.size(), args, "feval");
		    } else {
		    	char* command = mxArrayToString(mHandle);
			    exception = mexCallMATLABWithTrap(result.size(), result, args.size(), args, command);
			    mxFree(command);
		    }

		    if(exception != NULL) {
		        mxArray* message_pointer = mxGetProperty(exception, 0, "message");

		        if(message_pointer != NULL) {
		        	char* message = mxArrayToString(message_pointer);
		        	mexErrMsgIdAndTxt("MEX:Function:excute", "Error running MATLAB function:\n\t%s", message);
		        	mxFree(message);
			    } else {
		        	mexErrMsgIdAndTxt("MEX:Function:excute", "Error running MATLAB function\n\tUnknown Error!");
			    }
			    // Clean up
			    mxDestroyArray(exception);
		    }

		    return result;
		}

	private:
		const mxArray* mHandle;
        const mxClassID mClassID;
	};
}

#endif
