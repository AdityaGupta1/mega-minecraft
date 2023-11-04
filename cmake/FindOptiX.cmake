
find_path(
		OPTIX_INCLUDE_DIRS
		NAMES optix.h
		PATHS
		$ENV{OptiX}/include
		${OPTIX_ROOT_DIR}/include
		)
set(OPTIX_INCLUDE_DIRS ${OPTIX_INCLUDE_DIRS})
