cmake_minimum_required(VERSION 3.16)

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

set(ROCM_ENABLED OFF CACHE STRING "Build with ROCm/HIP support")
set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to the ROCm SDK")

set(IOM_SOURCES
        src/iom.cpp
        src/llama.cpp
)

set(IOM_HEADERS
        include/iom/iom.hpp
        include/iom/llama.hpp
)

if(ROCM_ENABLED)
    set(IOM_ROCM_ENABLED TRUE)
    list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")
    find_package(hip REQUIRED CONFIG)
    list(APPEND IOM_SOURCES src/rocm.cpp)
    list(APPEND IOM_HEADERS include/iom/rocm.hpp)
endif()

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

target_compile_features(libiom PUBLIC cxx_std_20)

target_link_libraries(libiom
    PRIVATE
        nlohmann_json::nlohmann_json
)

if(ROCM_ENABLED)
    target_compile_definitions(libiom PUBLIC IOM_ROCM_ENABLED=1)
    target_include_directories(libiom PUBLIC "${ROCM_PATH}/include")
    target_link_libraries(libec PUBLIC hip::host)
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
