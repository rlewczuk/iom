cmake_minimum_required(VERSION 3.21)

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

    target_compile_definitions(iom_ttnn
        PRIVATE
            IOM_TTNN_ENABLED=1
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
        src/cuda/driver.hpp
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

    target_compile_definitions(iom_cuda
        PRIVATE
            IOM_CUDA_ENABLED=1
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
            IOM_ROCM_ENABLED=1
            __HIP_PLATFORM_AMD__
    )
endif()

add_executable(iom
        src/main.cpp
)

target_link_libraries(iom
    PRIVATE
        libiom
)

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
