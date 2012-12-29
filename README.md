# MCM

A C++ implementation of conditional models such as the MCGSM.

## Requirements

* NumPy >= 2.6.0
* automake >= 1.11.0
* libtool >= 2.4.0

I have tested it with the above versions, but older versions might also work.

## Installation

### Linux

Make sure autoconf, automake and libtool are installed.

	apt-get install autoconf automake libtool

Go to `./code/liblbfgs` and execute the following:

	./autogen.sh
	./configure --enable-sse2
	make CFLAGS="-fPIC"

Once the L-BFGS library is compiled, go back to the root directory and execute:

	python setup.py build
	python setup.py install

### Mac OS X

First, make sure you have recent versions of automake and libtool installed. The versions that come
with Xcode 4.3 didn't work for me. You can use [Homebrew](http://mxcl.github.com/homebrew/) to install
newer ones.

	brew install autoconf automake libtool
	brew link autoconf automake libtool

Next, go to `./code/liblbfgs` and execute the following:

	./autogen.sh
	./configure --disable-dependency-tracking --enable-sse2
	make CFLAGS="-arch x86_64 -arch i386"

Once the L-BFGS library is compiled, go back to the root directory and execute:

	python setup.py build
	python setup.py install

### Building with the Intel compiler and MKL

To get even better performance, you might want to try compiling the module with Intel's compiler and
the MKL libraries. This probably requires some changes of the paths in `setup.py`. After that, use
the following line to compile the code

	python setup.py build --compiler=intelem

on 64-bit systems and

	python setup.py build --compiler=intel

on 32-bit systems.
