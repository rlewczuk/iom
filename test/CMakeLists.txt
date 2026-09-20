add_executable(iom_scalar_add_tests
        backend/test_scalar_add.cpp
)

target_include_directories(iom_scalar_add_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}
)

target_link_libraries(iom_scalar_add_tests
    PRIVATE
        doctest::doctest
        libiom
)

add_test(NAME iom_scalar_add_tests COMMAND iom_scalar_add_tests)
 
add_executable(iom_backend_conformance_sdpa_reference_tests
        backend/backend_conformance_sdpa_reference_test.cpp
)

set_source_files_properties(
        backend/backend_conformance_sdpa_reference_test.cpp PROPERTIES
        COMPILE_DEFINITIONS DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
)

target_include_directories(iom_backend_conformance_sdpa_reference_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}
)

target_link_libraries(iom_backend_conformance_sdpa_reference_tests
    PRIVATE
        doctest::doctest
        libiom
)

add_test(
        NAME iom_backend_conformance_sdpa_reference_tests
        COMMAND iom_backend_conformance_sdpa_reference_tests
)

add_executable(iom_rope_reference_tests
        backend/test_rope_reference.cpp
)

target_include_directories(iom_rope_reference_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}
)

target_link_libraries(iom_rope_reference_tests
    PRIVATE
        doctest::doctest
        libiom
)

add_test(NAME iom_rope_reference_tests COMMAND iom_rope_reference_tests)

add_executable(iom_scalar_silu_tests
        backend/test_scalar_silu.cpp
)

target_include_directories(iom_scalar_silu_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}
)

target_link_libraries(iom_scalar_silu_tests
    PRIVATE
        doctest::doctest
        libiom
)

add_test(NAME iom_scalar_silu_tests COMMAND iom_scalar_silu_tests)

# Opt-in real-checkpoint loading verification (change 0006-tinyllama /
# 02-configuration-weight-loading). Default OFF: an ordinary configuration
# compiles and runs the whole suite without any model artifact, hardcoded path,
# or download. When ON, each existing conformance target below compiles one
# additional `<Backend> real model loading` case that loads the caller-supplied
# official checkpoint directory named by the runtime `IOM_TEST_MODEL_DIR` and
# pinned `IOM_TEST_MODEL_ID`; the standard-GPU targets also read
# `IOM_TEST_MODEL_ARENA_BYTES`, because the synthetic arena cannot hold that
# reference. No new executable or test target is created.
option(IOM_TEST_REAL_MODEL_LOADING
    "Compile the opt-in real-checkpoint loading cases into the existing backend conformance tests"
    OFF
)

add_executable(iom_tests
        test_iom.cpp
        test_token_selection.cpp
        test_mmap.cpp
        test_safetensors.cpp
        test_tokenizer.cpp
        test_model_loading.cpp
        test_chat_format.cpp
        test_alloc.cpp
        test_gpu_status.cpp
)
set_source_files_properties(test_iom.cpp PROPERTIES
    COMPILE_DEFINITIONS DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
)

target_link_libraries(iom_tests
    PRIVATE
        doctest::doctest
        libiom
)

add_test(NAME iom_tests COMMAND iom_tests)

add_executable(iom_cpu_tests
        cpu/test_cpu.cpp
)

set_source_files_properties(cpu/test_cpu.cpp PROPERTIES
    COMPILE_DEFINITIONS DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
)

target_include_directories(iom_cpu_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(iom_cpu_tests
    PRIVATE
        doctest::doctest
        libiom
)

add_test(NAME iom_cpu_tests COMMAND iom_cpu_tests)

add_executable(iom_cpu_bench
        cpu/test_cpu_bench.cpp
)

target_link_libraries(iom_cpu_bench
    PRIVATE
        doctest::doctest
        libiom
)

set_source_files_properties(cpu/test_cpu_bench.cpp PROPERTIES
    COMPILE_DEFINITIONS DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
)

# CPU bandwidth/latency benchmark. Registered as a generic CTest test, the
# executable reports workload, compiler/OS, detected hardware, and sample
# distributions without failing on the one-host absolute floors. Threshold
# enforcement is a deliberate controlled-run contract: the binary asserts the
# calibrated floors/allowance only when the environment variable
# IOM_CPU_BENCH_ENFORCE_FLOORS=1 is set AND the detected runner matches the
# calibrated reference host (AMD Ryzen AI 9 HX 370, Linux x86-64). See
# docs/changes/01-review/17-PF-006-decouple-cpu-bench-from-ctest-thresholds.
add_test(NAME iom_cpu_bench COMMAND iom_cpu_bench)

function(add_iom_backend_tests name)
    cmake_parse_arguments(PARSE_ARGV 1 arg
        ""
        "BACKEND_LIBRARY;CONFORMANCE_DOCTEST_CONFIG"
        "SMOKE_MAIN_SOURCE;CONFORMANCE_MAIN_SOURCE;SMOKE_SOURCES;CONFORMANCE_SOURCES;COMPILE_DEFINITIONS;SMOKE_INCLUDE_DIRS;CONFORMANCE_INCLUDE_DIRS;COMPILE_OPTIONS;LINK_OPTIONS;RUNTIME_LIBRARIES")

    if(NOT DEFINED arg_CONFORMANCE_DOCTEST_CONFIG)
        set(arg_CONFORMANCE_DOCTEST_CONFIG DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN)
    endif()

    add_executable(iom_${name}_smoke_tests
        ${arg_SMOKE_SOURCES}
    )
    add_executable(iom_${name}_conformance_tests
        ${arg_CONFORMANCE_SOURCES}
    )

    set_source_files_properties(${arg_SMOKE_MAIN_SOURCE} PROPERTIES
        COMPILE_DEFINITIONS DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
    )
    set_source_files_properties(${arg_CONFORMANCE_MAIN_SOURCE} PROPERTIES
        COMPILE_DEFINITIONS ${arg_CONFORMANCE_DOCTEST_CONFIG}
    )

    if(arg_COMPILE_DEFINITIONS)
        target_compile_definitions(iom_${name}_smoke_tests
            PRIVATE
                ${arg_COMPILE_DEFINITIONS}
        )
        target_compile_definitions(iom_${name}_conformance_tests
            PRIVATE
                ${arg_COMPILE_DEFINITIONS}
        )
    endif()

    # The opt-in real-checkpoint loading case is compiled into the enabled
    # conformance executable only, so an OFF configuration and every smoke
    # target stay artifact-free.
    if(IOM_TEST_REAL_MODEL_LOADING)
        target_compile_definitions(iom_${name}_conformance_tests
            PRIVATE
                IOM_TEST_REAL_MODEL_LOADING=1
        )
    endif()

    if(arg_SMOKE_INCLUDE_DIRS)
        target_include_directories(iom_${name}_smoke_tests
            PRIVATE
                ${arg_SMOKE_INCLUDE_DIRS}
        )
    endif()
    if(arg_CONFORMANCE_INCLUDE_DIRS)
        target_include_directories(iom_${name}_conformance_tests
            PRIVATE
                ${arg_CONFORMANCE_INCLUDE_DIRS}
        )
    endif()

    if(arg_COMPILE_OPTIONS)
        target_compile_options(iom_${name}_smoke_tests
            PRIVATE
                ${arg_COMPILE_OPTIONS}
        )
        target_compile_options(iom_${name}_conformance_tests
            PRIVATE
                ${arg_COMPILE_OPTIONS}
        )
    endif()
    if(arg_LINK_OPTIONS)
        target_link_options(iom_${name}_smoke_tests
            PRIVATE
                ${arg_LINK_OPTIONS}
        )
        target_link_options(iom_${name}_conformance_tests
            PRIVATE
                ${arg_LINK_OPTIONS}
        )
    endif()

    target_link_libraries(iom_${name}_smoke_tests
        PRIVATE
            doctest::doctest
            ${arg_BACKEND_LIBRARY}
            libiom
    )
    target_link_libraries(iom_${name}_conformance_tests
        PRIVATE
            doctest::doctest
            ${arg_BACKEND_LIBRARY}
            libiom
    )

    if(arg_RUNTIME_LIBRARIES)
        target_link_libraries(iom_${name}_smoke_tests
            PRIVATE
                ${arg_RUNTIME_LIBRARIES}
        )
        target_link_libraries(iom_${name}_conformance_tests
            PRIVATE
                ${arg_RUNTIME_LIBRARIES}
        )
    endif()

    add_test(NAME iom_${name}_smoke_tests COMMAND iom_${name}_smoke_tests)
    add_test(NAME iom_${name}_conformance_tests COMMAND iom_${name}_conformance_tests)
endfunction()

if(ROCM_ENABLED)
    add_iom_backend_tests(rocm
        SMOKE_MAIN_SOURCE rocm/test_rocm_smoke.cpp
        CONFORMANCE_MAIN_SOURCE rocm/test_rocm_conformance.cpp
        SMOKE_SOURCES rocm/test_rocm_smoke.cpp
        CONFORMANCE_SOURCES rocm/test_rocm_conformance.cpp
        backend/backend_conformance_sdpa.hpp
        COMPILE_DEFINITIONS __HIP_PLATFORM_AMD__
        SMOKE_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src/rocm
        CONFORMANCE_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src
        BACKEND_LIBRARY iom_rocm
        RUNTIME_LIBRARIES hip::host
        CONFORMANCE_DOCTEST_CONFIG DOCTEST_CONFIG_IMPLEMENT
    )
endif()

if(CUDA_ENABLED)
    add_iom_backend_tests(cuda
        SMOKE_MAIN_SOURCE cuda/test_cuda_smoke.cpp
        CONFORMANCE_MAIN_SOURCE cuda/test_cuda_conformance.cpp
        SMOKE_SOURCES cuda/test_cuda_smoke.cpp
        CONFORMANCE_SOURCES cuda/test_cuda_conformance.cpp
        backend/backend_conformance_sdpa.hpp
        SMOKE_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src/cuda
        CONFORMANCE_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src
        BACKEND_LIBRARY iom_cuda
        RUNTIME_LIBRARIES CUDA::cuda_driver
    )
    add_executable(iom_cuda_sdpa_nonmatrix_tests
        cuda/test_cuda_sdpa_nonmatrix.cpp
    )
    target_include_directories(iom_cuda_sdpa_nonmatrix_tests
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${PROJECT_SOURCE_DIR}
            ${PROJECT_SOURCE_DIR}/src
    )
    target_link_libraries(iom_cuda_sdpa_nonmatrix_tests
        PRIVATE
            doctest::doctest
            iom_cuda
            libiom
            CUDA::cuda_driver
            CUDA::cudart
    )
    add_test(
        NAME iom_cuda_sdpa_nonmatrix_tests
        COMMAND iom_cuda_sdpa_nonmatrix_tests
    )
endif()


if(SYCL_ENABLED)
    add_iom_backend_tests(sycl
        SMOKE_MAIN_SOURCE sycl/test_sycl_smoke.cpp
        CONFORMANCE_MAIN_SOURCE sycl/test_sycl_conformance.cpp
        SMOKE_SOURCES sycl/test_sycl_smoke.cpp
        CONFORMANCE_SOURCES sycl/test_sycl_conformance.cpp
        backend/backend_conformance_sdpa.hpp
        SMOKE_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src/sycl ${IOM_SYCL_INCLUDE_DIR}
        CONFORMANCE_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/src/sycl ${IOM_SYCL_INCLUDE_DIR}
        COMPILE_DEFINITIONS IOM_ENABLE_TESTING
        # The conformance oracle's scalar recurrence runs in the host process,
        # whose FP environment the compiler's default model sets to FTZ/DAZ;
        # the contract forbids fast-math and flush-to-zero, so this target
        # keeps IEEE FP32 subnormals and NaN/infinity classes observable.
        COMPILE_OPTIONS -fsycl -ffp-model=precise
        LINK_OPTIONS -fsycl
        BACKEND_LIBRARY iom_sycl
        RUNTIME_LIBRARIES
    )
endif()


add_executable(iom_backend_conformance_cpu_tests
        cpu/test_cpu_conformance.cpp
        backend/backend_conformance_sdpa.hpp
)
set_source_files_properties(cpu/test_cpu_conformance.cpp PROPERTIES
    COMPILE_DEFINITIONS DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
)

target_include_directories(iom_backend_conformance_cpu_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(iom_backend_conformance_cpu_tests
    PRIVATE
        doctest::doctest
        libiom
)

add_test(NAME iom_backend_conformance_cpu_tests COMMAND iom_backend_conformance_cpu_tests)

# The CPU conformance executable is created directly rather than through
# `add_iom_backend_tests`, so the opt-in real-checkpoint loading case is enabled
# on it here with the same option.
if(IOM_TEST_REAL_MODEL_LOADING)
    target_compile_definitions(iom_backend_conformance_cpu_tests
        PRIVATE
            IOM_TEST_REAL_MODEL_LOADING=1
    )
endif()


# Combined coexistence executable (change 0001-tensor-view / 15): one
# translation unit includes every enabled public factory header and links
# exactly the enabled backend libraries next to each other. Each
# IOM_COEXIST_* macro gates that backend's runtime participation, so the
# same target serves every option combination. With no accelerator
# enabled there is nothing to compose beyond the CPU conformance target
# above, so the target exists only when at least one backend is enabled.
if(CUDA_ENABLED OR ROCM_ENABLED OR SYCL_ENABLED)
    add_executable(iom_backend_coexistence_tests
            backend/test_backend_coexistence.cpp
    )

    if(SYCL_ENABLED AND ROCM_ENABLED)
        # A HIP object in the link graph otherwise makes CMake select
        # clang-linker-wrapper, which cannot link DPC++ SYCL objects with
        # oneAPI SDKs that do not ship libLLVMSYCL.so. Keep the final link
        # on the DPC++ driver so it uses the installed libsycl runtime.
        set_property(TARGET iom_backend_coexistence_tests
            PROPERTY LINKER_LANGUAGE CXX
        )
    endif()

    set_source_files_properties(backend/test_backend_coexistence.cpp PROPERTIES
        COMPILE_DEFINITIONS DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
    )

    target_include_directories(iom_backend_coexistence_tests
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
    )

    target_link_libraries(iom_backend_coexistence_tests
        PRIVATE
            doctest::doctest
            libiom
    )

    if(CUDA_ENABLED)
        target_compile_definitions(iom_backend_coexistence_tests
            PRIVATE
                IOM_COEXIST_CUDA=1
        )

        target_link_libraries(iom_backend_coexistence_tests
            PRIVATE
                iom_cuda
                CUDA::cuda_driver
        )
    endif()

    if(ROCM_ENABLED)
        target_compile_definitions(iom_backend_coexistence_tests
            PRIVATE
                IOM_COEXIST_ROCM=1
                __HIP_PLATFORM_AMD__
        )

        target_link_libraries(iom_backend_coexistence_tests
            PRIVATE
                iom_rocm
                hip::host
        )
    endif()


    if(SYCL_ENABLED)
        target_compile_definitions(iom_backend_coexistence_tests
            PRIVATE
                IOM_COEXIST_SYCL=1
        )

        target_include_directories(iom_backend_coexistence_tests
            PRIVATE
                ${PROJECT_SOURCE_DIR}/src/sycl
                ${IOM_SYCL_INCLUDE_DIR}
        )

        target_compile_options(iom_backend_coexistence_tests
            PRIVATE
                -fsycl
        )

        target_link_options(iom_backend_coexistence_tests
            PRIVATE
                -fsycl
        )

        target_link_libraries(iom_backend_coexistence_tests
            PRIVATE
                iom_sycl
        )
    endif()

    add_test(NAME iom_backend_coexistence_tests
             COMMAND iom_backend_coexistence_tests)
endif()
