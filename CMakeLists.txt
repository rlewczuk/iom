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

set(IOM_SOURCES
        src/iom.cpp
        src/llama.cpp
        src/mmap.cpp
        src/safetensors.cpp
        src/alloc.cpp
)

set(IOM_HEADERS
        include/iom/iom.hpp
        include/iom/llama.hpp
        include/iom/mmap.hpp
        include/iom/safetensors.hpp
        include/iom/alloc.hpp
        include/iom/tensor.hpp
)

set(IOM_KERNEL_SOURCES
)

if(CUDA_ENABLED AND ROCM_ENABLED)
    message(FATAL_ERROR "CUDA_ENABLED and ROCM_ENABLED cannot both be enabled")
elseif(CUDA_ENABLED)
    set(CMAKE_CUDA_COMPILER "${CUDA_PATH}/bin/nvcc")
    enable_language(CUDA)
    set(IOM_KERNEL_LANGUAGE CUDA)
elseif(ROCM_ENABLED)
    file(STRINGS "${ROCM_PATH}/.info/version" IOM_ROCM_VERSION LIMIT_COUNT 1)
    if(IOM_ROCM_VERSION VERSION_LESS 7.2)
        message(FATAL_ERROR
            "ROCm 7.2 or newer is required for WMMA; detected ${IOM_ROCM_VERSION}"
        )
    endif()

    set(CMAKE_HIP_COMPILER_ROCM_ROOT "${ROCM_PATH}")
    set(CMAKE_HIP_COMPILER "${ROCM_PATH}/llvm/bin/clang++")
    enable_language(HIP)
    set(IOM_KERNEL_LANGUAGE HIP)
endif()

if(IOM_KERNEL_LANGUAGE AND IOM_KERNEL_SOURCES)
    set_source_files_properties(${IOM_KERNEL_SOURCES} PROPERTIES
        LANGUAGE "${IOM_KERNEL_LANGUAGE}"
    )
    list(APPEND IOM_SOURCES ${IOM_KERNEL_SOURCES})
endif()

add_library(libiom STATIC
        ${IOM_SOURCES}
        ${IOM_HEADERS}
)

set_target_properties(libiom PROPERTIES
    OUTPUT_NAME iom
)

if(IOM_KERNEL_LANGUAGE)
    set_target_properties(libiom PROPERTIES
        "${IOM_KERNEL_LANGUAGE}_STANDARD" 20
        "${IOM_KERNEL_LANGUAGE}_STANDARD_REQUIRED" ON
    )
endif()

target_include_directories(libiom
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

if(CUDA_ENABLED)
    target_compile_definitions(libiom PUBLIC IOM_CUDA_ENABLED=1)
elseif(ROCM_ENABLED)
    target_compile_definitions(libiom PUBLIC IOM_ROCM_ENABLED=1)
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
