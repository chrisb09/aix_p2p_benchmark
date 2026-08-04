cmake_minimum_required(VERSION 3.20)
project(aix_p2p_benchmark LANGUAGES CXX)

option(WITH_SCOREP "Compile manual Score-P regions" OFF)
set(AIXELERATOR_INSTALL_PREFIX "" CACHE PATH "AIxeleratorService installation prefix")

if(NOT AIXELERATOR_INSTALL_PREFIX)
    message(FATAL_ERROR "Set AIXELERATOR_INSTALL_PREFIX to an AIxeleratorService installation.")
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_library(AIXELERATOR_LIBRARY
    NAMES AIxeleratorService
    HINTS "${AIXELERATOR_INSTALL_PREFIX}/lib"
    REQUIRED)

add_executable(aix_p2p_benchmark main.cpp)
target_compile_features(aix_p2p_benchmark PRIVATE cxx_std_17)
target_include_directories(aix_p2p_benchmark PRIVATE "${AIXELERATOR_INSTALL_PREFIX}/include")
target_link_libraries(aix_p2p_benchmark PRIVATE MPI::MPI_CXX "${AIXELERATOR_LIBRARY}")
set_target_properties(aix_p2p_benchmark PROPERTIES
    BUILD_RPATH "${AIXELERATOR_INSTALL_PREFIX}/lib"
    INSTALL_RPATH "${AIXELERATOR_INSTALL_PREFIX}/lib")

# Score-P-enabled AIx links NVML. On CPU-only build hosts, resolve its symbols
# against CUDA's stub; GPU ranks still load the host's real driver at runtime.
if(DEFINED ENV{CUDA_ROOT} AND EXISTS "$ENV{CUDA_ROOT}/targets/x86_64-linux/lib/stubs/libnvidia-ml.so")
    target_link_directories(aix_p2p_benchmark PRIVATE "$ENV{CUDA_ROOT}/targets/x86_64-linux/lib/stubs")
    target_link_libraries(aix_p2p_benchmark PRIVATE nvidia-ml)
endif()

if(WITH_SCOREP)
    target_compile_definitions(aix_p2p_benchmark PRIVATE USE_SCOREP)
endif()
