#
# Find OptiX
#
# Try to find OptiX : NVIDIA OptiX SDK.
# This module defines 
# - OPTIX_INCLUDE_DIRS
# - OPTIX_FOUND
#
# The following variables can be set as arguments for the module.
# - OPTIX_ROOT_DIR : Root library directory of GLM 
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
		GLM_INCLUDE_DIR
		NAMES glm/glm.hpp
		PATHS
		/usr/include
		/usr/local/include
		/sw/include
		/opt/local/include
		${OPTIX_ROOT_DIR}/include
		DOC "The directory where optix.h resides")
endif()

# Handle REQUIRD argument, define *_FOUND variable
find_package_handle_standard_args(OPTIX DEFAULT_MSG OPTIX_INCLUDE_DIR)

# Define OPTIX_INCLUDE_DIRS
if (OPTIX_FOUND)
	set(OPTIX_INCLUDE_DIRS ${OPTIX_INCLUDE_DIR})
endif()

# Hide some variables
mark_as_advanced(OPTIX_INCLUDE_DIR)
