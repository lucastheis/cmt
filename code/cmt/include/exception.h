#ifndef EXCEPTION_H
#define EXCEPTION_H

class Exception {
	public:
		inline Exception(const char* message = "");

		inline const char* message();

	protected:
		const char* mMessage;
};



inline Exception::Exception(const char* message) : mMessage(message) {
}



inline const char* Exception::message() {
	return mMessage;
}

#endif
