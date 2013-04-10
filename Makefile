# compiler and linker options
CC = \
	$(shell python -c "import sysconfig; print(sysconfig.get_config_vars('CC')[0]);")
CFLAGS = -Wno-write-strings \
	$(shell python -c "import sysconfig; print(sysconfig.get_config_vars('CFLAGS')[0]);")
LD = $(CC)
LDFLAGS = \
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
	$(SRCDIR)/callbacktrain.cpp \
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
OBJECTS = $(patsubst %,$(OBJDIR)/%,$(SOURCES:.cpp=.o)) code/liblbfgs/lib/.libs/liblbfgs.a

MODULE = $(OBJDIR)/cmt.so

# keep object files around
.SECONDARY:

all: $(MODULE)

clean:
	rm $(OBJECTS)

install: $(MODULE)
	@cp $(MODULE) $(PYTHONPATH)

%.so: $(OBJECTS) 
	@echo $(LD) $(LDFLAGS) -o $@
	@$(LD) $(OBJECTS) $(LDFLAGS) -o $@

$(OBJDIR)/%.o: %.cpp
	@mkdir -p $(@D)
	@echo $(CC) -o $@ -c $^
	@$(CC) $(INCLUDE) $(CFLAGS) -o $@ -c $^
