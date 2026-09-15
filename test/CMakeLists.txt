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

add_executable(iom_tests
        test_iom.cpp
        test_mmap.cpp
        test_safetensors.cpp
        test_model_loading.cpp
        test_alloc.cpp
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
        SMOKE_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src/cuda
        CONFORMANCE_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src
        BACKEND_LIBRARY iom_cuda
        RUNTIME_LIBRARIES CUDA::cuda_driver
    )
endif()

if(SYCL_ENABLED)
    add_iom_backend_tests(sycl
        SMOKE_MAIN_SOURCE sycl/test_sycl_smoke.cpp
        CONFORMANCE_MAIN_SOURCE sycl/test_sycl_conformance.cpp
        SMOKE_SOURCES sycl/test_sycl_smoke.cpp
        CONFORMANCE_SOURCES sycl/test_sycl_conformance.cpp
        SMOKE_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src/sycl ${IOM_SYCL_INCLUDE_DIR}
        CONFORMANCE_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/src/sycl ${IOM_SYCL_INCLUDE_DIR}
        COMPILE_DEFINITIONS IOM_ENABLE_TESTING
        COMPILE_OPTIONS -fsycl
        LINK_OPTIONS -fsycl
        BACKEND_LIBRARY iom_sycl
        RUNTIME_LIBRARIES
    )
endif()


add_executable(iom_backend_conformance_cpu_tests
        cpu/test_cpu_conformance.cpp
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

if(TTNN_ENABLED)
    add_iom_backend_tests(ttnn
        SMOKE_MAIN_SOURCE ttnn/test_ttnn_smoke.cpp
        CONFORMANCE_MAIN_SOURCE ttnn/test_ttnn_conformance.cpp
        SMOKE_SOURCES ttnn/test_ttnn_smoke.cpp
        CONFORMANCE_SOURCES ttnn/test_ttnn_conformance.cpp
        CONFORMANCE_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        BACKEND_LIBRARY iom_ttnn
        RUNTIME_LIBRARIES TT::Metalium TTNN::TTNN
    )
endif()

# Combined coexistence executable (change 0001-tensor-view / 15): one
# translation unit includes every enabled public factory header and links
# exactly the enabled backend libraries next to each other. Each
# IOM_COEXIST_* macro gates that backend's runtime participation, so the
# same target serves every option combination. With no accelerator
# enabled there is nothing to compose beyond the CPU conformance target
# above, so the target exists only when at least one backend is enabled.
if(CUDA_ENABLED OR ROCM_ENABLED OR TTNN_ENABLED OR SYCL_ENABLED)
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

    if(TTNN_ENABLED)
        target_compile_definitions(iom_backend_coexistence_tests
            PRIVATE
                IOM_COEXIST_TTNN=1
        )

        target_link_libraries(iom_backend_coexistence_tests
            PRIVATE
                iom_ttnn
                TT::Metalium
                TTNN::TTNN
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
