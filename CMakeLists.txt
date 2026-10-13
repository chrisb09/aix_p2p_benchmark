cmake_minimum_required(VERSION 3.20)
project(aix_p2p_benchmark LANGUAGES CXX)

option(WITH_SCOREP "Compile manual Score-P regions" OFF)
option(WITH_TORCH_MODEL "Enable real TorchScript model mode in the staged fixture" OFF)
set(AIXELERATOR_INSTALL_PREFIX "" CACHE PATH "AIxeleratorService installation prefix")

find_package(MPI REQUIRED COMPONENTS CXX)

# This MPI-only fixture deliberately has no AIx/CUDA dependency. It measures the
# exact collective and credit-based input assembly patterns used by AIx.
add_executable(aix_input_assembly_mpi input_assembly_mpi.cpp)
target_compile_features(aix_input_assembly_mpi PRIVATE cxx_std_17)
target_link_libraries(aix_input_assembly_mpi PRIVATE MPI::MPI_CXX)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_executable(aix_input_assembly_staged input_assembly_staged.cu)
    target_compile_features(aix_input_assembly_staged PRIVATE cxx_std_17)
    target_link_libraries(aix_input_assembly_staged PRIVATE MPI::MPI_CXX CUDA::cudart)
    set_target_properties(aix_input_assembly_staged PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES 90)

    if(WITH_TORCH_MODEL)
        find_package(Torch REQUIRED)
        target_compile_definitions(aix_input_assembly_staged PRIVATE WITH_TORCH_MODEL)
        target_include_directories(aix_input_assembly_staged PRIVATE ${TORCH_INCLUDE_DIRS})
        separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
        target_compile_options(aix_input_assembly_staged PRIVATE ${TORCH_CXX_FLAGS_LIST})
        target_link_libraries(aix_input_assembly_staged PRIVATE ${TORCH_LIBRARIES})
        list(GET TORCH_LIBRARIES 0 TORCH_FIRST_LIBRARY)
        get_filename_component(TORCH_LIBRARY_DIR "${TORCH_FIRST_LIBRARY}" DIRECTORY)
        set_target_properties(aix_input_assembly_staged PROPERTIES
            BUILD_RPATH "${TORCH_LIBRARY_DIR}"
            INSTALL_RPATH "${TORCH_LIBRARY_DIR}")
    endif()
else()
    message(STATUS "CUDA compiler unavailable; skipping staged input assembly fixture")
endif()

if(AIXELERATOR_INSTALL_PREFIX)
    find_library(AIXELERATOR_LIBRARY
        NAMES AIxeleratorService
        HINTS "${AIXELERATOR_INSTALL_PREFIX}/lib"
        REQUIRED)

    add_executable(aix_p2p_benchmark main.cpp)
    target_compile_features(aix_p2p_benchmark PRIVATE cxx_std_17)
    target_include_directories(aix_p2p_benchmark PRIVATE "${AIXELERATOR_INSTALL_PREFIX}/include")
    target_link_libraries(aix_p2p_benchmark PRIVATE MPI::MPI_CXX "${AIXELERATOR_LIBRARY}" ${CMAKE_DL_LIBS})
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
else()
    message(STATUS "AIXELERATOR_INSTALL_PREFIX not set; building MPI-only input assembly fixture only")
endif()
