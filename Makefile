OS = $(shell uname)

INCDIR = code/cmt/include
SRCDIR = code/cmt/src
PYIDIR = code/cmt/python/include
PYSDIR = code/cmt/python/src
OBJDIR = build

# compiler and linker options
ifeq ($(OS), Darwin)
CXX = \
	$(shell python -c "import sysconfig; print(sysconfig.get_config_vars('CC')[0]);")
CXXFLAGS = $(shell python -c "import sysconfig; print(sysconfig.get_config_vars('CFLAGS')[0]);") \
	-Wno-write-strings -Wno-sign-compare -Wno-unknown-pragmas -Wno-parentheses -DEIGEN_NO_DEBUG
LD = $(CXX)
LDFLAGS = code/liblbfgs/lib/.libs/liblbfgs.a \
	$(shell python -c "import sysconfig; print(' '.join(sysconfig.get_config_vars('LDSHARED')[0].split(' ')[1:]));")
else
CXX = \
	$(shell python -c "import sysconfig; print(sysconfig.get_config_vars('CXX')[0]);")
CXXFLAGS = $(shell python -c "import sysconfig; print(sysconfig.get_config_vars('CFLAGS')[0]);") \
	-std=c++0x -Wno-write-strings -Wno-sign-compare -Wno-unknown-pragmas -Wno-parentheses -Wno-cpp -fPIC -fopenmp -DEIGEN_NO_DEBUG
LD = $(CXX)
LDFLAGS = code/liblbfgs/lib/.libs/liblbfgs.a -lgomp \
	$(shell python -c "import sysconfig; print(' '.join(sysconfig.get_config_vars('LDSHARED')[0].split(' ')[1:]));")
endif


# include paths
INCPYTHON = \
	$(shell python -c "from distutils import sysconfig; print(sysconfig.get_python_inc());")
INCNUMPY = \
	$(shell python -c "import os; from numpy.distutils import misc_util; print(os.path.join(misc_util.get_numpy_include_dirs()[0], 'numpy'));")
INCLUDE = -Icode -I$(INCDIR) -I$(PYIDIR) -Icode/liblbfgs/include -I$(INCPYTHON) -I$(INCNUMPY)

PYTHONPATH = \
	$(shell python -c "from distutils import sysconfig; print(sysconfig.get_python_lib());")

# source and object files
SOURCES = \
	$(SRCDIR)/affinepreconditioner.cpp \
	$(SRCDIR)/affinetransform.cpp \
	$(SRCDIR)/binningtransform.cpp \
	$(PYSDIR)/callbackinterface.cpp \
	$(SRCDIR)/conditionaldistribution.cpp \
	$(PYSDIR)/conditionaldistributioninterface.cpp \
	$(SRCDIR)/distribution.cpp \
	$(PYSDIR)/distributioninterface.cpp \
	$(PYSDIR)/fvbninterface.cpp \
	$(SRCDIR)/gsm.cpp \
	$(PYSDIR)/gsminterface.cpp \
	$(SRCDIR)/glm.cpp \
	$(PYSDIR)/glminterface.cpp \
	$(SRCDIR)/mcgsm.cpp \
	$(PYSDIR)/mcgsminterface.cpp \
	$(SRCDIR)/mcbm.cpp \
	$(PYSDIR)/mcbminterface.cpp \
	$(SRCDIR)/mixture.cpp \
	$(PYSDIR)/mixtureinterface.cpp \
	$(SRCDIR)/mlr.cpp \
	$(PYSDIR)/mlrinterface.cpp \
	$(PYSDIR)/module.cpp \
	$(SRCDIR)/nonlinearities.cpp \
	$(PYSDIR)/nonlinearitiesinterface.cpp \
	$(SRCDIR)/patchmodel.cpp \
	$(PYSDIR)/patchmodelinterface.cpp \
	$(SRCDIR)/pcapreconditioner.cpp \
	$(SRCDIR)/pcatransform.cpp \
	$(SRCDIR)/preconditioner.cpp \
	$(PYSDIR)/preconditionerinterface.cpp \
	$(PYSDIR)/pyutils.cpp \
	$(SRCDIR)/regularizer.cpp \
	$(SRCDIR)/stm.cpp \
	$(PYSDIR)/stminterface.cpp \
	$(SRCDIR)/tools.cpp \
	$(PYSDIR)/toolsinterface.cpp \
	$(SRCDIR)/trainable.cpp \
	$(PYSDIR)/trainableinterface.cpp \
	$(SRCDIR)/utils.cpp \
	$(SRCDIR)/univariatedistributions.cpp \
	$(PYSDIR)/univariatedistributionsinterface.cpp \
	$(SRCDIR)/whiteningpreconditioner.cpp \
	$(SRCDIR)/whiteningtransform.cpp
OBJECTS = $(patsubst %.cpp,$(OBJDIR)/%.o,$(SOURCES))

MODULE = $(OBJDIR)/_cmt.so

# keep object files around
.SECONDARY:

all: $(MODULE)

clean:
	rm -f $(OBJECTS) $(OBJECTS:.o=.d) $(MODULE)

install: $(MODULE)
	cp $(MODULE) $(PYTHONPATH)

$(MODULE): $(OBJECTS)
	@echo $(LD) $(LDFLAGS) -o $@
	@$(LD) $(OBJECTS) $(LDFLAGS) -o $@

$(OBJDIR)/%.o: %.cpp $(OBJDIR)/%.d
	@mkdir -p $(@D)
	@echo $(CXX) -o $@ -c $<
	@$(CXX) $(INCLUDE) $(CXXFLAGS) -o $@ -c $<

$(OBJDIR)/%.d: %.cpp
	@mkdir -p $(@D)
	@echo $(CXX) -MM $< -MF $@
	@$(CXX) $(INCLUDE) $(CXXFLAGS) -MM -MT '$(@:.d=.o)' $< -MF $@

-include $(OBJECTS:.o=.d)
