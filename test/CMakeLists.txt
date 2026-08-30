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
        COMPILE_DEFINITIONS DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
    )

    target_compile_definitions(iom_rocm_conformance_tests
        PRIVATE
            __HIP_PLATFORM_AMD__
    )

    target_include_directories(iom_rocm_conformance_tests
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
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
