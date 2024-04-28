cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
# The project name must match the name in setup.py.
project(_dda_model LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Eigen and Pybind11 dependencies are installed as git submodules.
add_subdirectory(deps/pybind11 EXCLUDE_FROM_ALL)
add_subdirectory(deps/eigen EXCLUDE_FROM_ALL)

include_directories(deps)
include_directories(deps/eigen)

# CUDA headers are taken from the system (e.g., in a conda
# environment, run `conda install cuda -c nvidia`)
find_package(CUDAToolkit)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# The Python headers are also taken from the system. If we are 
# running within an environment (e.g., conda), the following line
# makes sure we find the correct Python installation.
set(Python3_FIND_VIRTUALENV "STANDARD")
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# We need the -fPIC compilation flag because the C++ library will be
# linked into the Python extension module.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# We want to be able to include headers from within the src/ directory 
# without specifying full paths.
add_subdirectory(src)
include_directories(${PROJECT_SOURCE_DIR}/src)


################################################################################
# List of all C++ and CUDA source files to include in the DDA Model library.
################################################################################
set(ALL_SOURCES
    ${PROJECT_SOURCE_DIR}/src/AProductCore.cpp
    # ${PROJECT_SOURCE_DIR}/src/CoreStructure.cpp
    ${PROJECT_SOURCE_DIR}/src/DDAModel.cpp
    # ${PROJECT_SOURCE_DIR}/src/EvoDDAModel.cpp
    ${PROJECT_SOURCE_DIR}/src/FilterOption.cpp
    # ${PROJECT_SOURCE_DIR}/src/filterReader.cpp
    # ${PROJECT_SOURCE_DIR}/src/FOM.cpp
    # ${PROJECT_SOURCE_DIR}/src/main.cpp
    ${PROJECT_SOURCE_DIR}/src/ObjDDAModel.cpp
    # ${PROJECT_SOURCE_DIR}/src/ObjReader.cpp
    ${PROJECT_SOURCE_DIR}/src/SiCi.cpp
    # ${PROJECT_SOURCE_DIR}/src/symReader.cpp
    ${PROJECT_SOURCE_DIR}/src/Tools.cpp
    ${PROJECT_SOURCE_DIR}/src/Kernel.cu
)
source_group("sources" FILES ${ALL_SOURCES})

################################################################################
# Build the DDA Model library (to link from C++ executables / Python bindings).
################################################################################

add_library(_dda_model_lib STATIC ${ALL_SOURCES})

set_property(TARGET _dda_model_lib PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_property(TARGET _dda_model_lib PROPERTY CUDA_ARCHITECTURES OFF)

################################################################################
# Build the Python bindings. 
################################################################################

pybind11_add_module(_dda_model ${PROJECT_SOURCE_DIR}/python_bindings/PythonBindings.cpp)
set_property(TARGET _dda_model PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_property(TARGET _dda_model PROPERTY CUDA_ARCHITECTURES OFF)
target_link_libraries(_dda_model PUBLIC _dda_model_lib cufft Eigen3::Eigen)

################################################################################
# Build the Python bindings. 
################################################################################


# Build the main (CPP exe).

# add_executable(main main.cpp)
# set_property(TARGET main PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
# set_property(TARGET main PROPERTY CUDA_ARCHITECTURES OFF)
# target_link_libraries(main PUBLIC dda_model_lib cufft Eigen3::Eigen)

# Build the Python bindings.

#-------------------------------------------------------------------#
# Print out some information about the configuration
#-------------------------------------------------------------------#
message("===================================================")
message("  ${PROJECT_NAME} build configuration:")
message("===================================================")
message("  CUDA compiler ID      : ${CMAKE_CUDA_COMPILER_ID}")
message("  CUDA compiler Version : ${CMAKE_CUDA_COMPILER_VERSION}")
message("  C++ Compiler : ${CMAKE_CXX_COMPILER_ID} "
  "${CMAKE_CXX_COMPILER_VERSION} "
  "${CMAKE_CXX_COMPILER_WRAPPER}")
message("    ${CMAKE_CXX_COMPILER}")
message("  CUDA Compiler      : ${CMAKE_CUDA_COMPILER}")
message("  CUDA Compiler exec : ${CUDA_NVCC_EXECUTABLE}")
message("  CUDA Compile flags : ${CMAKE_CUDA_FLAGS}")
message("  CUDA toolkit inc   : ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
message("")
message("  Python3_found       : ${Python3_FOUND}")
message("  Python3 version     : ${Python3_VERSION}")
message("  Python3 interpreter : ${Python3_EXECUTABLE}")
message("  Python3 interp. Id  : ${Python3_INTERPRETER_ID}")
message("  Python3_INCLUDE_DIRS: ${Python3_INCLUDE_DIRS}")
message("  Python3_LIBRARY_DIRS: ${Python3_LIBRARY_DIRS}")


