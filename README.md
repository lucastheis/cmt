# Conditional Modeling Toolkit

![samples](https://raw.github.com/lucastheis/cmt/develop/media/samples.jpg)

Fast implementations of several probabilistic models. Examples:

* MCGSM (mixture of conditional Gaussian scale mixtures; Theis et al., 2012)
* MCBM (mixture of conditional Boltzmann machines)
* FVBN (fully-visible belief network; Neal, 1992)
* GLM (generalized linear model; Nelder & Wedderburn, 1972)
* MLR (multinomial logistic regression)
* STM (spike-triggered mixture model; Theis et al., 2013)

## Requirements

* Python >= 2.7.0
* NumPy >= 1.6.1
* automake >= 1.11.0
* libtool >= 2.4.0

I have tested the code with the above versions, but older versions might also work.

## Example

```python
from cmt import MCGSM, WhiteningPreconditioner

# load data
input, output = load('data')

# preprocessing
wt = WhiteningPreconditioner(input, output)

# fit a conditional model to predict outputs from inputs
model = MCGSM(
	dim_in=input.shape[0],
	dim_out=output.shape[0],
	num_components=8,
	num_scales=6,
	num_features=40)
model.initialize(*wt(input, output))
model.train(*wt(input, output), parameters={
	'max_iter': 1000,
	'threshold': 1e-5})

# evaluate log-likelihood [nats] on the training data
loglik = model.loglikelihood(*wt(input, output)) + wt.logjacobian(input, output)
```

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

on 32-bit systems. The following might be helpful when trying to compile the L-BFGS library with the
Intel compiler.

	./autogen.sh
	CC=icc ./configure --enable-sse2
	CC=icc make CFLAGS="-fPIC"
