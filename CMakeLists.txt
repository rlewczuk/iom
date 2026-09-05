cmake_minimum_required(VERSION 3.21)
option(SYCL_ENABLED "Build with SYCL support" OFF)
set(SYCL_COMPILER "" CACHE FILEPATH
    "SYCL compiler executable (icpx or dpcpp)")

if(SYCL_ENABLED)
    if(SYCL_COMPILER)
        if(NOT EXISTS "${SYCL_COMPILER}")
            message(FATAL_ERROR
                "SYCL compiler does not exist: ${SYCL_COMPILER}")
        endif()
        set(IOM_SYCL_COMPILER "${SYCL_COMPILER}")
    else()
        find_program(IOM_SYCL_COMPILER NAMES icpx dpcpp)
    endif()

    if(NOT IOM_SYCL_COMPILER)
        message(FATAL_ERROR
            "SYCL_ENABLED=ON requires the Intel oneAPI DPC++ compiler "
            "(icpx or dpcpp) in PATH; set SYCL_COMPILER to its executable")
    endif()

    get_filename_component(IOM_SYCL_COMPILER_BIN
        "${IOM_SYCL_COMPILER}" DIRECTORY)
    get_filename_component(IOM_SYCL_ROOT
        "${IOM_SYCL_COMPILER_BIN}/.." ABSOLUTE)
    find_path(IOM_SYCL_INCLUDE_DIR
        NAMES sycl/sycl.hpp
        HINTS "${IOM_SYCL_ROOT}/include")
    find_library(IOM_SYCL_RUNTIME_LIBRARY
        NAMES sycl
        HINTS "${IOM_SYCL_ROOT}/lib")
    if(NOT IOM_SYCL_INCLUDE_DIR OR NOT IOM_SYCL_RUNTIME_LIBRARY)
        message(FATAL_ERROR
            "SYCL_ENABLED=ON requires sycl/sycl.hpp and libsycl; "
            "searched under ${IOM_SYCL_ROOT}")
    endif()

    if(DEFINED CMAKE_CXX_COMPILER)
        get_filename_component(IOM_CONFIGURED_CXX_NAME
            "${CMAKE_CXX_COMPILER}" NAME)
        if(NOT IOM_CONFIGURED_CXX_NAME STREQUAL "icpx"
                AND NOT IOM_CONFIGURED_CXX_NAME STREQUAL "dpcpp")
            message(FATAL_ERROR
                "SYCL_ENABLED=ON requires an Intel oneAPI DPC++ compiler "
                "(icpx or dpcpp); the configured compiler is "
                "${CMAKE_CXX_COMPILER}")
        endif()
    else()
        set(CMAKE_CXX_COMPILER "${IOM_SYCL_COMPILER}" CACHE FILEPATH
            "C++ compiler selected for SYCL-enabled builds")
    endif()
endif()

project(iom
    VERSION 0.0.1
    DESCRIPTION "Inference of Oversized Models"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(nlohmann_json REQUIRED)

option(CUDA_ENABLED "Build with CUDA support" OFF)
set(CUDA_PATH "/usr/local/cuda" CACHE PATH "Path to the CUDA Toolkit")
option(ROCM_ENABLED "Build with ROCm support" OFF)
set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to the ROCm SDK")
option(TTNN_ENABLED "Build with TTNN support" OFF)

set(IOM_SOURCES
        src/iom.cpp
        src/cpu/device.cpp
        src/llama.cpp
        src/mmap.cpp
        src/safetensors.cpp
        src/alloc.cpp
)

set(IOM_HEADERS
        include/iom/iom.hpp
        include/iom/gpu_algorithm.hpp
        include/iom/cpu/device.hpp
        include/iom/llama.hpp
        include/iom/mmap.hpp
        include/iom/safetensors.hpp
        include/iom/alloc.hpp
        include/iom/device.hpp
        include/iom/tensor.hpp
)


add_library(libiom STATIC
        ${IOM_SOURCES}
        ${IOM_HEADERS}
)

set_target_properties(libiom PROPERTIES
    OUTPUT_NAME iom
)


target_include_directories(libiom
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(libiom
    PUBLIC
        nlohmann_json::nlohmann_json
)

if(TTNN_ENABLED)
    find_package(tt-nn CONFIG REQUIRED)
    if(NOT TARGET TT::Metalium OR NOT TARGET TTNN::TTNN)
        message(FATAL_ERROR
                "TTNN SDK is missing required TT::Metalium or TTNN::TTNN targets")
    endif()

    add_library(iom_ttnn STATIC
        src/ttnn/device.cpp
        src/ttnn/copy.cpp
        src/ttnn/copy.hpp
        include/iom/ttnn/device.hpp
    )

    target_include_directories(iom_ttnn
        PUBLIC
            ${PROJECT_SOURCE_DIR}/include
    )

    target_link_libraries(iom_ttnn
        PRIVATE
            libiom
            TT::Metalium
            TTNN::TTNN
    )

endif()
if(CUDA_ENABLED)
    set(CUDAToolkit_ROOT "${CUDA_PATH}")
    find_package(CUDAToolkit REQUIRED MODULE)
    enable_language(CUDA)
    add_library(iom_cuda STATIC
        src/cuda/device.cpp
        src/cuda/copy.cu
        src/cuda/copy.hpp
        src/cuda/driver.cpp
        src/cuda/driver.hpp
        src/cuda/transfer_pool.hpp
        src/shared/standard_tiled_copy.hpp
        include/iom/cuda/device.hpp
    )

    set_target_properties(iom_cuda PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
    )

    target_include_directories(iom_cuda
        PUBLIC
            ${PROJECT_SOURCE_DIR}/include
    )

    target_link_libraries(iom_cuda
        PRIVATE
            libiom
            CUDA::cuda_driver
            CUDA::cudart
    )

endif()
if(ROCM_ENABLED)
    # Config mode discovers the configured SDK and exposes hip::host for the
    # backend target. Check the package version separately so newer major
    # ROCm releases remain valid while preserving the 7.2 minimum.
    find_package(hip CONFIG REQUIRED
        PATHS "${ROCM_PATH}"
        NO_DEFAULT_PATH
    )
    if(NOT DEFINED hip_VERSION OR hip_VERSION VERSION_LESS 7.2)
        message(FATAL_ERROR
            "ROCm 7.2 or newer is required for WMMA; detected ${hip_VERSION}"
        )
    endif()
    enable_language(HIP)

    add_library(iom_rocm STATIC
        src/rocm/device.cpp
        src/rocm/copy.hip
        src/rocm/copy.hpp
        src/shared/standard_tiled_copy.hpp
        include/iom/rocm/device.hpp
    )

    target_include_directories(iom_rocm
        PUBLIC
            ${PROJECT_SOURCE_DIR}/include
    )

    target_link_libraries(iom_rocm
        PRIVATE
            libiom
            hip::host
    )

    target_compile_definitions(iom_rocm
        PRIVATE
            __HIP_PLATFORM_AMD__
    )

    # Unoptimized HIP device images retain the runtime assertion/hostcall
    # support even for kernels that do not use it. APUs without PCIe atomics
    # reject those images at dispatch, so keep debug and unconfigured builds
    # debuggable while applying the minimum optimization that removes the
    # unreachable hostcall path.
    target_compile_options(iom_rocm
        PRIVATE
            $<$<AND:$<COMPILE_LANGUAGE:HIP>,$<NOT:$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>,$<CONFIG:MinSizeRel>>>>:-O1>
    )
endif()
if(SYCL_ENABLED)
    add_library(iom_sycl STATIC
        src/sycl/device.cpp
        src/sycl/copy.cpp
        src/sycl/runtime.hpp
        src/sycl/copy.hpp
        include/iom/sycl/device.hpp
    )

    target_include_directories(iom_sycl
        PUBLIC
            ${PROJECT_SOURCE_DIR}/include
        PRIVATE
            ${PROJECT_SOURCE_DIR}/src/sycl
            ${IOM_SYCL_INCLUDE_DIR}
    )

    target_link_libraries(iom_sycl
        PRIVATE
            libiom
    )

    target_compile_options(iom_sycl
        PRIVATE
            -fsycl
    )

    target_link_options(iom_sycl
        PRIVATE
            -fsycl
    )

endif()


include(CTest)

if(BUILD_TESTING)
  find_package(doctest CONFIG QUIET)
  if(NOT doctest_FOUND)
    message(FATAL_ERROR
            "doctest was not found. On Ubuntu, install it with:\n"
            "  sudo apt update && sudo apt install doctest-dev\n"
            "Then rerun CMake."
        )
  endif()

  add_subdirectory(test)
endif()
