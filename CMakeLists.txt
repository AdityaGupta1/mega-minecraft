cmake_minimum_required(VERSION 3.17)
project(mega_minecraft LANGUAGES CXX CUDA)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Enable C++11 for host code
set(CMAKE_CXX_STANDARD 11)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

########################################
# CUDA Setup
########################################
find_package(CUDAToolkit 10 REQUIRED)
include(${CMAKE_MODULE_PATH}/CUDAComputesList.cmake)

#list(APPEND CUDA_NVCC_FLAGS ${CUDA_GENERATE_CODE})
#list(APPEND CUDA_NVCC_FLAGS_DEBUG "-g -G")
#set(CUDA_VERBOSE_BUILD ON)

include_directories(${CUDAToolkit_INCLUDE_DIRS})
message(STATUS ${CUDA_LIBRARIES})
message(STATUS ${CUDAToolkit_LIBRARY_DIR})
set(LIBRARIES CUDA::cudart)

########################################


find_package(OpenGL REQUIRED)

if(UNIX)
    find_package(glfw3 REQUIRED)
    find_package(GLEW REQUIRED)
    list(APPEND LIBRARIES cuda glfw ${GLEW_LIBRARIES} ${OPENGL_gl_LIBRARY})
else(UNIX)
    set(EXTERNAL "external")

    set(GLFW_ROOT_DIR ${EXTERNAL})
    set(GLFW_USE_STATIC_LIBS ON)
    find_package(GLFW REQUIRED)

    set(GLEW_ROOT_DIR ${EXTERNAL})
    set(GLEW_USE_STATIC_LIBS ON)
    find_package(GLEW REQUIRED)

    add_definitions(${GLEW_DEFINITIONS})
    include_directories(${GLEW_INCLUDE_DIR} ${GLFW_INCLUDE_DIR})
    list(APPEND LIBRARIES cuda ${GLEW_LIBRARY} ${GLFW_LIBRARY} ${OPENGL_LIBRARY})
endif(UNIX)

set(GLM_ROOT_DIR "external")
set(OPTIX_ROOT_DIR "external")

find_package(GLM REQUIRED)
include_directories(${GLM_INCLUDE_DIRS})
message(STATUS "GLM include directories: ${GLM_INCLUDE_DIRS}")

include_directories(${CMAKE_SOURCE_DIR}/src)

########################################
# OptiX Setup + Compile
########################################
find_package(OPTIX REQUIRED)

include_directories(${OPTIX_INCLUDE_DIRS})
include("nvcuda_compile_module")

file(GLOB_RECURSE SHADERS "src/shaders/*.cu")
file(GLOB_RECURSE SHADER_HEADERS "src/shaders/*.h" "src/shaders/*.hpp")

# When using OptiX SDK 7.5.0 and CUDA 11.7 or higher, the modules can either be built from OptiX IR input or from PTX input.
# OPTIX_PROGRAM_TARGET and OPTIX_MODULE_EXTENSION switch the NVCC compilation between the two options.
NVCUDA_COMPILE_MODULE(
  SOURCES ${SHADERS}
  DEPENDENCIES ${SHADERS_HEADERS}
  TARGET_PATH "${CMAKE_CURRENT_BINARY_DIR}/optix_shaders"
  EXTENSION "${OPTIX_MODULE_EXTENSION}"
  GENERATED_FILES PROGRAM_MODULES
  NVCC_OPTIONS "${OPTIX_PROGRAM_TARGET}" "--machine=64" "--gpu-architecture=compute_50" "--use_fast_math" "--relocatable-device-code=true" "--generate-line-info" "-Wno-deprecated-gpu-targets" "-I${OPTIX_INCLUDE_DIR}" "-I${CMAKE_CURRENT_SOURCE_DIR}/shaders"
)

# Generate a header file with the file names
set(SHADER_LIST_FILE "${CMAKE_CURRENT_BINARY_DIR}/optix_shaders/ShaderList.h")
file(WRITE ${SHADER_LIST_FILE} "#pragma once\n#include <vector>\n#include <string>\nconst std::vector<std::string> shaderFiles = {\n")
foreach(SHADER_FILE ${PROGRAM_MODULES})
    file(APPEND ${SHADER_LIST_FILE} "    \"${SHADER_FILE}\",\n")
endforeach()
file(APPEND ${SHADER_LIST_FILE} "};\n")

# Add the binary dir to the include directories so that ShaderList.h can be included
include_directories(${CMAKE_CURRENT_BINARY_DIR}/optix_shaders)

if(USE_OPTIX_IR)
add_definitions(
  # This define switches the OptiX program module filenames to either *.optixir or *.ptx extensions at compile time.
  "-DUSE_OPTIX_IR"
)
endif()

########################################

file(GLOB_RECURSE headers "src/*.h" "src/*.hpp")
file(GLOB_RECURSE sources "src/*.cpp" "src/*.cu")

list(SORT headers)
list(SORT sources)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src" PREFIX "Headers" FILES ${headers})
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src" PREFIX "Sources" FILES ${sources})

foreach(SHADER_SRC ${SHADERS})
    set_source_files_properties(${SHADER_SRC} PROPERTIES HEADER_FILE_ONLY TRUE)
endforeach()

add_executable( ${CMAKE_PROJECT_NAME}
  ${headers}
  ${sources}
  ${PROGRAM_MODULES}
)

message(STATUS ${COMPUTES_DETECTED_LIST})
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "${COMPUTES_DETECTED_LIST}")

target_link_libraries(${CMAKE_PROJECT_NAME} ${LIBRARIES})

set(SHADER_DIR ${CMAKE_SOURCE_DIR}/shaders)
set(SHADER_OUTPUT_DIR ${CMAKE_BINARY_DIR}/shaders)

add_custom_command(
    OUTPUT ${SHADER_OUTPUT_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${SHADER_DIR}
        ${SHADER_OUTPUT_DIR}
    COMMENT "Copying shaders from ${SHADER_DIR} to ${SHADER_OUTPUT_DIR}"
    DEPENDS DIRECTORY ${SHADER_DIR}
)

add_custom_target(
    CopyShadersTarget ALL
    DEPENDS ${SHADER_OUTPUT_DIR}
)

set(TEXTURE_DIR ${CMAKE_SOURCE_DIR}/textures)
set(TEXTURE_OUTPUT_DIR ${CMAKE_BINARY_DIR}/textures)

add_custom_command(
    OUTPUT ${TEXTURE_OUTPUT_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${TEXTURE_DIR}
        ${TEXTURE_OUTPUT_DIR}
    COMMENT "Copying textures from ${TEXTURE_DIR} to ${TEXTURE_OUTPUT_DIR}"
    DEPENDS DIRECTORY ${TEXTURE_DIR}
)

add_custom_target(
    CopyTexturesTarget ALL
    DEPENDS ${TEXTURE_OUTPUT_DIR}
)

add_dependencies(${CMAKE_PROJECT_NAME} CopyShadersTarget CopyTexturesTarget)

