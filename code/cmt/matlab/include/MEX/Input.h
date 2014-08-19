#ifndef __MEX_INPUT_H__
#define __MEX_INPUT_H__

#include "mex.h"

#include "ObjectHandle.h"

#include "Eigen/Core"
using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::ColMajor;

namespace MEX {
	class Function;

	namespace Type{
		enum Type { // ToDo: Decide if you want to add vectors
		        Unknown,
		        BoolScalar,
		        IntScalar,
		        FloatScalar,
		        BoolMatrix,
		        IntMatrix,
		        FloatMatrix,
		        String,
		        Struct,
		        Cell,
		        Function
		};
	}

	class Input {
	public:
	    Input(int size, const mxArray** data);

	    int size() const;

	    bool has(int i) const;

	    class Getter {
	    public:
	        Getter(const mxArray* data, int index);

	        bool isEmpty();

	        Type::Type getType();

	        bool isType(MEX::Type::Type t);

	        template<class BaseClass> BaseClass* unwrap() {
            	return 	ObjectHandle<BaseClass>::unwrap(mData);
	        }

	        operator MatrixXd ();

	        operator VectorXd ();

	        operator double ();

	        operator std::string ();

	        operator int ();

	        operator bool ();

	        operator Function ();

	    private:
	        const mxArray* mData;
	        int mIndex; // Only needed for error messages
	    };

	    Input::Getter operator[](int i) const;

	    template<class StructClass>
	    StructClass toStruct(unsigned int offset, bool (*parser)(StructClass*, std::string, Input::Getter)) const {

	        StructClass params;

	        // ToDo: Add support for matlab structs here!

	        if((mSize - offset) % 2 != 0) {
	            mexErrMsgIdAndTxt("mexWrapper:unpairedOptions", "Options must consist of name-value pairs.");
	        }

	        for(; offset < mSize; offset += 2) {
	            std::string key = Getter(mData[offset], offset);

	            if(parser(&params, key, Getter(mData[offset + 1], offset + 1))) {
	                continue;
	            } else {
	                mexErrMsgIdAndTxt("mexWrapper:unknownOption", "Unknown option: '%s'", key.c_str());
	            }
	        }

	        return params;
	    }


	private:
		// ToDo: Disable copy operation to avoid confusion?!
        //Input(const Input& other);
        //Input & operator=(const Input&);

	    int mSize;
	    const mxArray** mData;
	};
}

#endif
