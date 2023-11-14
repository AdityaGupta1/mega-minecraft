#
# Find OptiX
#
# Try to find OptiX : NVIDIA OptiX SDK.
# This module defines 
# - OPTIX_INCLUDE_DIRS
# - OPTIX_FOUND
#
# The following variables can be set as arguments for the module.
# - OPTIX_ROOT_DIR : Root library directory of OptiX
#
#

# Additional modules
include(FindPackageHandleStandardArgs)

if (WIN32)
	# Find include files
	find_path(
		OPTIX_INCLUDE_DIR
		NAMES optix.h
		PATHS
		$ENV{OptiX}/include
		$ENV{OptiX_SDK}/include
		${OPTIX_ROOT_DIR}/include
		"C:/ProgramData/NVIDIA Corporation/OptiX SDK *"
		DOC "The directory where optix.h resides")
else()
	# Find include files
	find_path(
		OPTIX_INCLUDE_DIR
		NAMES optix.h
		PATHS
		$ENV{OptiX}/include
		$ENV{OptiX_SDK}/include
		${OPTIX_ROOT_DIR}/include
		"/usr/NVIDIA-OptiX-SDK-*-linux64/include"
		"/usr/local/NVIDIA-OptiX-SDK-*-linux64/include"
		"/sw/NVIDIA-OptiX-SDK-*-linux64/include"
		"/opt/local/NVIDIA-OptiX-SDK-*-linux64/include"
		${OPTIX_ROOT_DIR}/include
		~"/NVIDIA-OptiX-SDK-*-linux64/include"
		DOC "The directory where optix.h resides")
endif()

# Handle REQUIRD argument, define *_FOUND variable
find_package_handle_standard_args(OPTIX DEFAULT_MSG OPTIX_INCLUDE_DIR)

# Define OPTIX_INCLUDE_DIRS
if (OPTIX_FOUND)
	set(OPTIX_INCLUDE_DIRS ${OPTIX_INCLUDE_DIR})
	string(REGEX REPLACE ".*OptiX[ -]SDK[ -]([0-9]\\.[0-9])\\..*" "\\1" OPTIX_VERSION ${OPTIX_INCLUDE_DIR})
	message(STATUS "OptiX Version ${OPTIX_VERSION}")
	if(${OPTIX_VERSION} VERSION_GREATER_EQUAL "7.5")
		set(USE_OPTIX_IR TRUE)
		set(OPTIX_MODULE_EXTENSION ".optixir")	
		set(OPTIX_PROGRAM_TARGET "--optix-ir")
	else()
		set(USE_OPTIX_IR FALSE)
		set(OPTIX_MODULE_EXTENSION ".ptx")
		set(OPTIX_PROGRAM_TARGET "--ptx")
	endif()
endif()

# Hide some variables
mark_as_advanced(OPTIX_INCLUDE_DIR)