project(cmt CXX)
cmake_minimum_required(VERSION 2.8)

set(CMT_VERSION_MAJOR 0 )
set(CMT_VERSION_MINOR 1 )


# Find matlab
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

find_package(MATLAB REQUIRED)
if(MATLAB_FOUND)
	message(STATUS "Matlab found at ${MATLAB_INCLUDE_DIR}")
else(MATLAB_FOUND)
	message(FATAL_ERROR "Matlab not found")
endif(MATLAB_FOUND)

include_directories(${MATLAB_INCLUDE_DIR})

## includes ##

# Matlab interface header
include_directories(include)

# CMT headers
include_directories(../include)

# Eigen headers
include_directories(../..)

# lbfgs headers
include_directories(../../liblbfgs/include)

## libaries ##

# lbfgs library
link_directories(${CMAKE_SOURCE_DIR}/../../liblbfgs/lib/.libs)

## files to compile ##

# cmt source
SET(CMT_SOURCE
	../src/affinepreconditioner.cpp
	../src/affinetransform.cpp
	../src/binningtransform.cpp
	../src/conditionaldistribution.cpp
	../src/distribution.cpp
	../src/gsm.cpp
	../src/glm.cpp
	../src/mcgsm.cpp
	../src/mcbm.cpp
	../src/mixture.cpp
	../src/mlr.cpp
	../src/nonlinearities.cpp
	../src/patchmodel.cpp
	../src/pcapreconditioner.cpp
	../src/pcatransform.cpp
	../src/preconditioner.cpp
	../src/regularizer.cpp
	../src/stm.cpp
	../src/tools.cpp
	../src/trainable.cpp
	../src/utils.cpp
	../src/univariatedistributions.cpp
	../src/whiteningpreconditioner.cpp
	../src/whiteningtransform.cpp
)

# common interface source
set(MEX_SOURCE
	src/conditionaldistributioninterface.cpp
	src/trainableinterface.cpp
)


ADD_DEFINITIONS(
	-std=c++0x
	-fvisibility=default
	-Bsymbolic
)

# make cmt library
add_library(cmt SHARED ${CMT_SOURCE})
target_link_libraries(cmt lbfgs)

set_target_properties(cmt PROPERTIES MACOSX_RPATH "on" INSTALL_RPATH "@loader_path")

# make mex files
add_library(glminterface SHARED src/glminterface.cpp ${MEX_SOURCE})
target_link_libraries(glminterface cmt ${MATLAB_LIBRARIES})


add_library(stminterface SHARED src/stminterface.cpp ${MEX_SOURCE})
target_link_libraries(stminterface cmt ${MATLAB_LIBRARIES})


set_target_properties(glminterface stminterface PROPERTIES SUFFIX ".mexmaci64" PREFIX "" MACOSX_RPATH "on")
