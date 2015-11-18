#ifndef CMT_EXCEPTION_H
#define CMT_EXCEPTION_H

#include <stdexcept>

namespace CMT {
	class Exception : public std::runtime_error {
		public:
			inline Exception(const char* message = "");

			inline const char* message();
	};
}

inline CMT::Exception::Exception(const char* message) : std::runtime_error(message) {}

inline const char* CMT::Exception::message() {
	return what();
}

#endif
