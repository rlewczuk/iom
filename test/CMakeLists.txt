add_executable(iom_tests
        test_iom.cpp
        test_mmap.cpp
        test_safetensors.cpp
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

add_test(NAME iom_cpu_bench COMMAND iom_cpu_bench)

if(ROCM_ENABLED)
    add_executable(iom_rocm_smoke_tests
            rocm/test_rocm_smoke.cpp
    )

    set_source_files_properties(rocm/test_rocm_smoke.cpp PROPERTIES
        COMPILE_DEFINITIONS DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
    )

    target_compile_definitions(iom_rocm_smoke_tests
        PRIVATE
            __HIP_PLATFORM_AMD__
    )
    target_include_directories(iom_rocm_smoke_tests
        PRIVATE
            ${PROJECT_SOURCE_DIR}/src/rocm
    )

    target_link_libraries(iom_rocm_smoke_tests
        PRIVATE
            doctest::doctest
            iom_rocm
            hip::host
    )

    add_test(NAME iom_rocm_smoke_tests COMMAND iom_rocm_smoke_tests)
    add_executable(iom_rocm_conformance_tests
            rocm/test_rocm_conformance.cpp
    )

    set_source_files_properties(rocm/test_rocm_conformance.cpp PROPERTIES
        COMPILE_DEFINITIONS DOCTEST_CONFIG_IMPLEMENT
    )

    target_compile_definitions(iom_rocm_conformance_tests
        PRIVATE
            __HIP_PLATFORM_AMD__
    )

    target_include_directories(iom_rocm_conformance_tests
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${PROJECT_SOURCE_DIR}/src
    )

    target_link_libraries(iom_rocm_conformance_tests
        PRIVATE
            doctest::doctest
            iom_rocm
            libiom
            hip::host
    )

    add_test(NAME iom_rocm_conformance_tests
             COMMAND iom_rocm_conformance_tests)
endif()

if(CUDA_ENABLED)
    add_executable(iom_cuda_smoke_tests
            cuda/test_cuda_smoke.cpp
    )

    set_source_files_properties(cuda/test_cuda_smoke.cpp PROPERTIES
        COMPILE_DEFINITIONS DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
    )


    target_include_directories(iom_cuda_smoke_tests
        PRIVATE
            ${PROJECT_SOURCE_DIR}/src/cuda
    )
    target_link_libraries(iom_cuda_smoke_tests
        PRIVATE
            doctest::doctest
            iom_cuda
            CUDA::cuda_driver
    )

    add_test(NAME iom_cuda_smoke_tests COMMAND iom_cuda_smoke_tests)
    add_executable(iom_cuda_conformance_tests
            cuda/test_cuda_conformance.cpp
    )

    set_source_files_properties(cuda/test_cuda_conformance.cpp PROPERTIES
        COMPILE_DEFINITIONS DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
    )

    target_include_directories(iom_cuda_conformance_tests
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${PROJECT_SOURCE_DIR}/src
    )

    target_link_libraries(iom_cuda_conformance_tests
        PRIVATE
            doctest::doctest
            iom_cuda
            libiom
            CUDA::cuda_driver
    )

    add_test(NAME iom_cuda_conformance_tests
             COMMAND iom_cuda_conformance_tests)
endif()


if(SYCL_ENABLED)
    add_executable(iom_sycl_smoke_tests
            sycl/test_sycl_smoke.cpp
    )

    set_source_files_properties(sycl/test_sycl_smoke.cpp PROPERTIES
        COMPILE_DEFINITIONS DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
    )

    target_include_directories(iom_sycl_smoke_tests
        PRIVATE
            ${PROJECT_SOURCE_DIR}/src/sycl
            ${IOM_SYCL_INCLUDE_DIR}
    )

    target_compile_options(iom_sycl_smoke_tests
        PRIVATE
            -fsycl
    )

    target_link_options(iom_sycl_smoke_tests
        PRIVATE
            -fsycl
    )

    target_link_libraries(iom_sycl_smoke_tests
        PRIVATE
            doctest::doctest
            iom_sycl
    )

    add_test(NAME iom_sycl_smoke_tests
             COMMAND iom_sycl_smoke_tests)

    add_executable(iom_sycl_conformance_tests
            sycl/test_sycl_conformance.cpp
    )

    set_source_files_properties(sycl/test_sycl_conformance.cpp PROPERTIES
        COMPILE_DEFINITIONS DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
    )

    target_include_directories(iom_sycl_conformance_tests
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${PROJECT_SOURCE_DIR}/src
            ${PROJECT_SOURCE_DIR}/src/sycl
            ${IOM_SYCL_INCLUDE_DIR}
    )

    target_compile_options(iom_sycl_conformance_tests
        PRIVATE
            -fsycl
    )

    target_link_options(iom_sycl_conformance_tests
        PRIVATE
            -fsycl
    )

    target_link_libraries(iom_sycl_conformance_tests
        PRIVATE
            doctest::doctest
            iom_sycl
            libiom
    )

    add_test(NAME iom_sycl_conformance_tests
             COMMAND iom_sycl_conformance_tests)
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
    add_executable(iom_ttnn_smoke_tests
            ttnn/test_ttnn_smoke.cpp
    )

    set_source_files_properties(ttnn/test_ttnn_smoke.cpp PROPERTIES
        COMPILE_DEFINITIONS DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
    )

    target_link_libraries(iom_ttnn_smoke_tests
        PRIVATE
            doctest::doctest
            iom_ttnn
            TT::Metalium
            TTNN::TTNN
    )

    add_test(NAME iom_ttnn_smoke_tests COMMAND iom_ttnn_smoke_tests)

    add_executable(iom_ttnn_conformance_tests
            ttnn/test_ttnn_conformance.cpp
    )

    set_source_files_properties(ttnn/test_ttnn_conformance.cpp PROPERTIES
        COMPILE_DEFINITIONS DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
    )

    target_include_directories(iom_ttnn_conformance_tests
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
    )

    target_link_libraries(iom_ttnn_conformance_tests
        PRIVATE
            doctest::doctest
            iom_ttnn
            libiom
            TT::Metalium
            TTNN::TTNN
    )

    add_test(NAME iom_ttnn_conformance_tests
             COMMAND iom_ttnn_conformance_tests)
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
