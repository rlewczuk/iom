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
        test_model_session.cpp
        test_alloc.cpp
        test_gpu_status.cpp
        test_inference_metrics.cpp
        test_model_reference_fixture.cpp
        test_model_cache_integration.cpp
        test_model_integration.cpp
        test_model_official_harness.cpp
)
set_source_files_properties(test_iom.cpp PROPERTIES
    COMPILE_DEFINITIONS DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
)

target_link_libraries(iom_tests
    PRIVATE
        doctest::doctest
        libiom
)
add_dependencies(iom_tests iom_generate)
target_compile_definitions(iom_tests
    PRIVATE
        IOM_GENERATE_EXECUTABLE="$<TARGET_FILE:iom_generate>"
        IOM_MODEL_REFERENCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/model"
)

add_test(NAME iom_tests COMMAND iom_tests)

find_program(IOM_PYTHON3_EXECUTABLE NAMES python3 REQUIRED)
add_test(
        NAME iom_official_wrapper_tests
        COMMAND
            "${CMAKE_COMMAND}" -E env PYTHONDONTWRITEBYTECODE=1
            "${IOM_PYTHON3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/model/test_run_official_inference.py"
)
set_tests_properties(iom_official_wrapper_tests PROPERTIES TIMEOUT 30)
add_test(
        NAME iom_official_wrapper_timeout_tests
        COMMAND
            "${CMAKE_COMMAND}" -E env PYTHONDONTWRITEBYTECODE=1
            "${IOM_PYTHON3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/model/test_run_official_inference.py"
            "WrapperRegressionTest.test_registered_timeout_reaps_child_and_publishes_evidence"
            "WrapperRegressionTest.test_registered_timeout_drains_residual_process_group"
)
set_tests_properties(
        iom_official_wrapper_timeout_tests PROPERTIES TIMEOUT 10
)

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
        "SMOKE_MAIN_SOURCE;CONFORMANCE_MAIN_SOURCE;SMOKE_SOURCES;CONFORMANCE_SOURCES;COMPILE_DEFINITIONS;SMOKE_INCLUDE_DIRS;CONFORMANCE_INCLUDE_DIRS;COMPILE_OPTIONS;LINK_OPTIONS;RUNTIME_LIBRARIES;CONFORMANCE_TEST_ARGUMENTS")

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
    # Optional per-backend CTest arguments keep an opt-in guarded case out of the artifact-free aggregate selection.
    add_test(NAME iom_${name}_conformance_tests
        COMMAND iom_${name}_conformance_tests ${arg_CONFORMANCE_TEST_ARGUMENTS})
endfunction()

# ROCm official TinyLlama inference evidence. This is an independent,
# default-off opt-in next to the ROCm conformance driver and the real-checkpoint
# loading option: an ordinary configuration needs no model artifact, no Python
# model library, and no GPU, while an enabled configuration must compile and
# register its real ROCm case instead of silently selecting nothing.
option(IOM_TEST_REAL_MODEL_INFERENCE_ROCM
    "Compile and register the opt-in official TinyLlama inference case for ROCm"
    OFF
)

if(IOM_TEST_REAL_MODEL_INFERENCE_ROCM AND NOT ROCM_ENABLED)
    message(FATAL_ERROR
        "IOM_TEST_REAL_MODEL_INFERENCE_ROCM requires ROCM_ENABLED=ON; the "
        "official ROCm case has no non-ROCm substitute"
    )
endif()

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
        # The guarded official case is compiled into this executable for the
        # wrapper only: the aggregate registration must never run it without
        # the wrapper's artifact, digest, and process-supervision checks.
        CONFORMANCE_TEST_ARGUMENTS "--test-case-exclude=ROCm real model inference"
    )
    target_compile_definitions(iom_rocm_conformance_tests
        PRIVATE
            IOM_MODEL_REFERENCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/model"
    )
    add_executable(iom_rocm_sdpa_nonmatrix_tests
        rocm/test_rocm_sdpa_nonmatrix.cpp
        backend/backend_conformance_sdpa_reference.hpp
    )
    set_source_files_properties(rocm/test_rocm_sdpa_nonmatrix.cpp PROPERTIES
        COMPILE_DEFINITIONS DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
    )
    target_include_directories(iom_rocm_sdpa_nonmatrix_tests
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${PROJECT_SOURCE_DIR}/src
            ${PROJECT_SOURCE_DIR}/src/rocm
    )
    target_link_libraries(iom_rocm_sdpa_nonmatrix_tests
        PRIVATE
            doctest::doctest
            iom_rocm
            libiom
            hip::host
    )
    add_test(
        NAME iom_rocm_sdpa_nonmatrix_tests
        COMMAND iom_rocm_sdpa_nonmatrix_tests
    )

    # Opt-in official TinyLlama inference: exactly one guarded Doctest case is
    # compiled into the existing ROCm conformance executable, and exactly
    # `iom_rocm_real_model_inference_tests` selects it through the Python
    # wrapper, which owns artifact validation, digest recomputation, bounded
    # process supervision, and evidence publication. The aggregate ROCm
    # registration excludes the case above and stays artifact-free.
    if(IOM_TEST_REAL_MODEL_INFERENCE_ROCM)
        find_package(Python3 REQUIRED COMPONENTS Interpreter)
        cmake_host_system_information(
            RESULT IOM_ROCM_OFFICIAL_BUILD_HOST QUERY HOSTNAME
        )
        cmake_host_system_information(
            RESULT IOM_ROCM_OFFICIAL_PROCESSOR_NAME QUERY PROCESSOR_NAME
        )
        cmake_host_system_information(
            RESULT IOM_ROCM_OFFICIAL_PROCESSOR_DESCRIPTION
            QUERY PROCESSOR_DESCRIPTION
        )
        string(TOUPPER
            "${CMAKE_BUILD_TYPE}"
            IOM_ROCM_OFFICIAL_BUILD_TYPE_UPPER
        )
        set(IOM_ROCM_OFFICIAL_CONFIG_FLAGS_VARIABLE
            "CMAKE_CXX_FLAGS_${IOM_ROCM_OFFICIAL_BUILD_TYPE_UPPER}"
        )
        set(IOM_ROCM_OFFICIAL_EFFECTIVE_CXX_FLAGS
            "${CMAKE_CXX_FLAGS} ${${IOM_ROCM_OFFICIAL_CONFIG_FLAGS_VARIABLE}}"
        )
        string(STRIP
            "${IOM_ROCM_OFFICIAL_EFFECTIVE_CXX_FLAGS}"
            IOM_ROCM_OFFICIAL_EFFECTIVE_CXX_FLAGS
        )
        target_compile_definitions(iom_rocm_conformance_tests
            PRIVATE
                IOM_TEST_REAL_MODEL_INFERENCE_ROCM=1
                IOM_OFFICIAL_BUILD_HOST="${IOM_ROCM_OFFICIAL_BUILD_HOST}"
                IOM_OFFICIAL_PROCESSOR_NAME="${IOM_ROCM_OFFICIAL_PROCESSOR_NAME}"
                IOM_OFFICIAL_PROCESSOR_DESCRIPTION="${IOM_ROCM_OFFICIAL_PROCESSOR_DESCRIPTION}"
                IOM_OFFICIAL_SYSTEM_PROCESSOR="${CMAKE_SYSTEM_PROCESSOR}"
                IOM_OFFICIAL_CXX_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}"
                IOM_OFFICIAL_CXX_COMPILER_VERSION="${CMAKE_CXX_COMPILER_VERSION}"
                IOM_OFFICIAL_BUILD_TYPE="$<CONFIG>"
                IOM_OFFICIAL_CXX_FLAGS="${IOM_ROCM_OFFICIAL_EFFECTIVE_CXX_FLAGS}"
        )
        add_test(
            NAME iom_rocm_real_model_inference_tests
            COMMAND
                "${CMAKE_COMMAND}" -E env PYTHONDONTWRITEBYTECODE=1
                "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/model/run_official_inference.py"
                --backend rocm
                --executable "$<TARGET_FILE:iom_rocm_conformance_tests>"
        )
        set_tests_properties(iom_rocm_real_model_inference_tests
            PROPERTIES
                LABELS "official-model"
                TIMEOUT 1800
        )
    endif()
endif()


if(CUDA_ENABLED)
    add_iom_backend_tests(cuda
        SMOKE_MAIN_SOURCE cuda/test_cuda_smoke.cpp
        CONFORMANCE_MAIN_SOURCE cuda/test_cuda_conformance.cpp
        SMOKE_SOURCES cuda/test_cuda_smoke.cpp
        CONFORMANCE_SOURCES cuda/test_cuda_conformance.cpp
        cuda/test_cuda_sdpa_matrix.cpp
        backend/backend_conformance_sdpa.hpp
        SMOKE_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src/cuda
        CONFORMANCE_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src
        BACKEND_LIBRARY iom_cuda
        RUNTIME_LIBRARIES CUDA::cuda_driver
        CONFORMANCE_TEST_ARGUMENTS "--test-case-exclude=CUDA real model inference"
    )
    target_compile_definitions(iom_cuda_conformance_tests
        PRIVATE
            IOM_MODEL_REFERENCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/model"
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

# CUDA official TinyLlama inference evidence. Like the CPU case this path is
# independent of IOM_TEST_REAL_MODEL_LOADING, defaults OFF, and never enters
# ordinary CTest: enabling it compiles the guarded case into the CUDA
# conformance binary, the aggregate registration keeps excluding that case, and
# exactly `iom_cuda_real_model_inference_tests` selects it through the Python
# wrapper that validates the caller artifacts and supervises the child.
option(IOM_TEST_REAL_MODEL_INFERENCE_CUDA
    "Compile and register the opt-in official TinyLlama inference case for CUDA"
    OFF
)

if(IOM_TEST_REAL_MODEL_INFERENCE_CUDA)
    if(NOT CUDA_ENABLED)
        message(FATAL_ERROR
            "IOM_TEST_REAL_MODEL_INFERENCE_CUDA requires CUDA_ENABLED=ON"
        )
    endif()
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    cmake_host_system_information(
        RESULT IOM_OFFICIAL_CUDA_BUILD_HOST QUERY HOSTNAME
    )
    cmake_host_system_information(
        RESULT IOM_OFFICIAL_CUDA_PROCESSOR_NAME QUERY PROCESSOR_NAME
    )
    cmake_host_system_information(
        RESULT IOM_OFFICIAL_CUDA_PROCESSOR_DESCRIPTION QUERY PROCESSOR_DESCRIPTION
    )
    string(TOUPPER "${CMAKE_BUILD_TYPE}" IOM_OFFICIAL_CUDA_BUILD_TYPE_UPPER)
    set(IOM_OFFICIAL_CUDA_CONFIG_FLAGS_VARIABLE
        "CMAKE_CXX_FLAGS_${IOM_OFFICIAL_CUDA_BUILD_TYPE_UPPER}"
    )
    set(IOM_OFFICIAL_CUDA_EFFECTIVE_CXX_FLAGS
        "${CMAKE_CXX_FLAGS} ${${IOM_OFFICIAL_CUDA_CONFIG_FLAGS_VARIABLE}}"
    )
    string(STRIP
        "${IOM_OFFICIAL_CUDA_EFFECTIVE_CXX_FLAGS}"
        IOM_OFFICIAL_CUDA_EFFECTIVE_CXX_FLAGS
    )
    target_compile_definitions(iom_cuda_conformance_tests
        PRIVATE
            IOM_TEST_REAL_MODEL_INFERENCE_CUDA=1
            IOM_OFFICIAL_BUILD_HOST="${IOM_OFFICIAL_CUDA_BUILD_HOST}"
            IOM_OFFICIAL_PROCESSOR_NAME="${IOM_OFFICIAL_CUDA_PROCESSOR_NAME}"
            IOM_OFFICIAL_PROCESSOR_DESCRIPTION="${IOM_OFFICIAL_CUDA_PROCESSOR_DESCRIPTION}"
            IOM_OFFICIAL_SYSTEM_PROCESSOR="${CMAKE_SYSTEM_PROCESSOR}"
            IOM_OFFICIAL_CXX_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}"
            IOM_OFFICIAL_CXX_COMPILER_VERSION="${CMAKE_CXX_COMPILER_VERSION}"
            IOM_OFFICIAL_BUILD_TYPE="$<CONFIG>"
            IOM_OFFICIAL_CXX_FLAGS="${IOM_OFFICIAL_CUDA_EFFECTIVE_CXX_FLAGS}"
    )
    add_test(
        NAME iom_cuda_real_model_inference_tests
        COMMAND
            "${CMAKE_COMMAND}" -E env PYTHONDONTWRITEBYTECODE=1
            "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/model/run_official_inference.py"
            --backend cuda
            --executable "$<TARGET_FILE:iom_cuda_conformance_tests>"
    )
    set_tests_properties(iom_cuda_real_model_inference_tests
        PROPERTIES
            LABELS "official-model"
            TIMEOUT 1800
    )
endif()

# SYCL-only official TinyLlama inference evidence. This remains independent
# of loading verification and is absent from ordinary CTest unless callers
# explicitly provide and enable the verified official artifacts.
option(IOM_TEST_REAL_MODEL_INFERENCE_SYCL
    "Compile and register the opt-in official TinyLlama inference case for SYCL"
    OFF
)
if(IOM_TEST_REAL_MODEL_INFERENCE_SYCL AND NOT SYCL_ENABLED)
    message(FATAL_ERROR
        "IOM_TEST_REAL_MODEL_INFERENCE_SYCL requires SYCL_ENABLED=ON"
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
        CONFORMANCE_TEST_ARGUMENTS "--test-case-exclude=SYCL real model inference"
    )
    target_compile_definitions(iom_sycl_conformance_tests
        PRIVATE
            IOM_MODEL_REFERENCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/model"
    )

    if(IOM_TEST_REAL_MODEL_INFERENCE_SYCL)
        find_package(Python3 REQUIRED COMPONENTS Interpreter)
        cmake_host_system_information(
            RESULT IOM_OFFICIAL_BUILD_HOST QUERY HOSTNAME
        )
        cmake_host_system_information(
            RESULT IOM_OFFICIAL_PROCESSOR_NAME QUERY PROCESSOR_NAME
        )
        cmake_host_system_information(
            RESULT IOM_OFFICIAL_PROCESSOR_DESCRIPTION QUERY PROCESSOR_DESCRIPTION
        )
        string(TOUPPER "${CMAKE_BUILD_TYPE}" IOM_OFFICIAL_BUILD_TYPE_UPPER)
        set(IOM_OFFICIAL_CONFIG_FLAGS_VARIABLE
            "CMAKE_CXX_FLAGS_${IOM_OFFICIAL_BUILD_TYPE_UPPER}"
        )
        set(IOM_OFFICIAL_EFFECTIVE_CXX_FLAGS
            "${CMAKE_CXX_FLAGS} ${${IOM_OFFICIAL_CONFIG_FLAGS_VARIABLE}}"
        )
        string(STRIP
            "${IOM_OFFICIAL_EFFECTIVE_CXX_FLAGS}"
            IOM_OFFICIAL_EFFECTIVE_CXX_FLAGS
        )
        target_compile_definitions(iom_sycl_conformance_tests
            PRIVATE
                IOM_TEST_REAL_MODEL_INFERENCE_SYCL=1
                IOM_OFFICIAL_BUILD_HOST="${IOM_OFFICIAL_BUILD_HOST}"
                IOM_OFFICIAL_PROCESSOR_NAME="${IOM_OFFICIAL_PROCESSOR_NAME}"
                IOM_OFFICIAL_PROCESSOR_DESCRIPTION="${IOM_OFFICIAL_PROCESSOR_DESCRIPTION}"
                IOM_OFFICIAL_SYSTEM_PROCESSOR="${CMAKE_SYSTEM_PROCESSOR}"
                IOM_OFFICIAL_CXX_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}"
                IOM_OFFICIAL_CXX_COMPILER_VERSION="${CMAKE_CXX_COMPILER_VERSION}"
                IOM_OFFICIAL_BUILD_TYPE="$<CONFIG>"
                IOM_OFFICIAL_CXX_FLAGS="${IOM_OFFICIAL_EFFECTIVE_CXX_FLAGS}"
        )
        add_test(
            NAME iom_sycl_real_model_inference_tests
            COMMAND
                "${CMAKE_COMMAND}" -E env PYTHONDONTWRITEBYTECODE=1
                "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/model/run_official_inference.py"
                --backend sycl
                --executable "$<TARGET_FILE:iom_sycl_conformance_tests>"
        )
        set_tests_properties(iom_sycl_real_model_inference_tests
            PROPERTIES
                LABELS "official-model"
                TIMEOUT 1800
        )
    endif()
endif()


# CPU-only official TinyLlama inference evidence. This remains independent of
# IOM_TEST_REAL_MODEL_LOADING and is absent from ordinary CTest unless callers
# explicitly provide and enable the verified official artifacts.
option(IOM_TEST_REAL_MODEL_INFERENCE_CPU
    "Compile and register the opt-in official TinyLlama inference case for CPU"
    OFF
)

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

# The official case is compiled into this binary only for the opt-in wrapper,
# but the ordinary aggregate registration must never bypass that wrapper's
# artifact validation, digest recomputation, or process supervision.
add_test(
        NAME iom_backend_conformance_cpu_tests
        COMMAND iom_backend_conformance_cpu_tests
                "--test-case-exclude=CPU real model inference"
)

# The CPU conformance executable is created directly rather than through
# `add_iom_backend_tests`, so the opt-in real-checkpoint loading case is enabled
# on it here with the same option.
if(IOM_TEST_REAL_MODEL_LOADING)
    target_compile_definitions(iom_backend_conformance_cpu_tests
        PRIVATE
            IOM_TEST_REAL_MODEL_LOADING=1
    )
endif()

if(IOM_TEST_REAL_MODEL_INFERENCE_CPU)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    cmake_host_system_information(
        RESULT IOM_OFFICIAL_BUILD_HOST QUERY HOSTNAME
    )
    cmake_host_system_information(
        RESULT IOM_OFFICIAL_PROCESSOR_NAME QUERY PROCESSOR_NAME
    )
    cmake_host_system_information(
        RESULT IOM_OFFICIAL_PROCESSOR_DESCRIPTION QUERY PROCESSOR_DESCRIPTION
    )
    string(TOUPPER "${CMAKE_BUILD_TYPE}" IOM_OFFICIAL_BUILD_TYPE_UPPER)
    set(IOM_OFFICIAL_CONFIG_FLAGS_VARIABLE
        "CMAKE_CXX_FLAGS_${IOM_OFFICIAL_BUILD_TYPE_UPPER}"
    )
    set(IOM_OFFICIAL_EFFECTIVE_CXX_FLAGS
        "${CMAKE_CXX_FLAGS} ${${IOM_OFFICIAL_CONFIG_FLAGS_VARIABLE}}"
    )
    string(STRIP
        "${IOM_OFFICIAL_EFFECTIVE_CXX_FLAGS}"
        IOM_OFFICIAL_EFFECTIVE_CXX_FLAGS
    )
    target_compile_definitions(iom_backend_conformance_cpu_tests
        PRIVATE
            IOM_TEST_REAL_MODEL_INFERENCE_CPU=1
            IOM_OFFICIAL_BUILD_HOST="${IOM_OFFICIAL_BUILD_HOST}"
            IOM_OFFICIAL_PROCESSOR_NAME="${IOM_OFFICIAL_PROCESSOR_NAME}"
            IOM_OFFICIAL_PROCESSOR_DESCRIPTION="${IOM_OFFICIAL_PROCESSOR_DESCRIPTION}"
            IOM_OFFICIAL_SYSTEM_PROCESSOR="${CMAKE_SYSTEM_PROCESSOR}"
            IOM_OFFICIAL_CXX_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}"
            IOM_OFFICIAL_CXX_COMPILER_VERSION="${CMAKE_CXX_COMPILER_VERSION}"
            IOM_OFFICIAL_BUILD_TYPE="$<CONFIG>"
            IOM_OFFICIAL_CXX_FLAGS="${IOM_OFFICIAL_EFFECTIVE_CXX_FLAGS}"
    )
    add_test(
        NAME iom_cpu_real_model_inference_tests
        COMMAND
            "${CMAKE_COMMAND}" -E env PYTHONDONTWRITEBYTECODE=1
            "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/model/run_official_inference.py"
            --backend cpu
            --executable "$<TARGET_FILE:iom_backend_conformance_cpu_tests>"
    )
    set_tests_properties(iom_cpu_real_model_inference_tests
        PROPERTIES
            LABELS "official-model"
            TIMEOUT 3900
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
