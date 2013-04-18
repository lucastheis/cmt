OS = $(shell uname)

# compiler and linker options
ifeq ($(OS), Darwin)
CXX = \
	$(shell python -c "import sysconfig; print(sysconfig.get_config_vars('CC')[0]);")
CFLAGS = $(shell python -c "import sysconfig; print(sysconfig.get_config_vars('CFLAGS')[0]);") \
	-Wno-write-strings -Wno-sign-compare -Wno-unknown-pragmas -Wno-parentheses
else
CXX = \
	$(shell python -c "import sysconfig; print(sysconfig.get_config_vars('CXX')[0]);")
CFLAGS = $(shell python -c "import sysconfig; print(sysconfig.get_config_vars('CFLAGS')[0]);") \
	-Wno-write-strings -Wno-sign-compare -Wno-unknown-pragmas -Wno-parentheses -Wno-cpp -fPIC
endif

LD = $(CXX)
LDFLAGS = code/liblbfgs/lib/.libs/liblbfgs.a \
	$(shell python -c "import sysconfig; print(' '.join(sysconfig.get_config_vars('LDSHARED')[0].split(' ')[1:]));")

# include paths
INCPYTHON = \
	$(shell python -c "from distutils import sysconfig; print(sysconfig.get_python_inc());")
INCNUMPY = \
	$(shell python -c "import os; from numpy.distutils import misc_util; print(os.path.join(misc_util.get_numpy_include_dirs()[0], 'numpy'));")
INCLUDE = -Icode -Icode/cmt/include -Icode/liblbfgs/include -I$(INCPYTHON) -I$(INCNUMPY)

PYTHONPATH = \
	$(shell python -c "from distutils import sysconfig; print(sysconfig.get_python_lib());")

# source and object files
SRCDIR = code/cmt/src
OBJDIR = build
SOURCES = \
	$(SRCDIR)/affinepreconditioner.cpp \
	$(SRCDIR)/callbackinterface.cpp \
	$(SRCDIR)/conditionaldistribution.cpp \
	$(SRCDIR)/conditionaldistributioninterface.cpp \
	$(SRCDIR)/distribution.cpp \
	$(SRCDIR)/distributioninterface.cpp \
	$(SRCDIR)/mcgsm.cpp \
	$(SRCDIR)/mcgsminterface.cpp \
	$(SRCDIR)/mcbm.cpp \
	$(SRCDIR)/mcbminterface.cpp \
	$(SRCDIR)/module.cpp \
	$(SRCDIR)/pcapreconditioner.cpp \
	$(SRCDIR)/preconditionerinterface.cpp \
	$(SRCDIR)/pyutils.cpp \
	$(SRCDIR)/tools.cpp \
	$(SRCDIR)/toolsinterface.cpp \
	$(SRCDIR)/utils.cpp \
	$(SRCDIR)/whiteningpreconditioner.cpp
OBJECTS = $(patsubst %,$(OBJDIR)/%,$(SOURCES:.cpp=.o))

MODULE = $(OBJDIR)/cmt.so

# keep object files around
.SECONDARY:

all: $(MODULE)

clean:
	rm -f $(OBJECTS) $(MODULE)

install: $(MODULE)
	cp $(MODULE) $(PYTHONPATH)

$(MODULE): $(OBJECTS) 
	@echo $(LD) $(LDFLAGS) -o $@
	@$(LD) $(OBJECTS) $(LDFLAGS) -o $@

$(OBJDIR)/%.o: %.cpp
	@mkdir -p $(@D)
	@echo $(CXX) -o $@ -c $^
	@$(CXX) $(INCLUDE) $(CFLAGS) -o $@ -c $^
