#ifndef CMT_EXCEPTION_H
#define CMT_EXCEPTION_H

namespace CMT {
	class Exception {
		public:
			inline Exception(const char* message = "");

			inline const char* message();

		protected:
			const char* mMessage;
	};
}


inline CMT::Exception::Exception(const char* message) : mMessage(message) {
}



inline const char* CMT::Exception::message() {
	return mMessage;
}

#endif
