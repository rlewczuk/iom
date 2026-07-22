add_executable(iom_tests
        test_iom.cpp
)

set_source_files_properties(test_iom.cpp PROPERTIES
    COMPILE_DEFINITIONS DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
)

if(IOM_ROCM_ENABLED)
    target_sources(iom_tests PRIVATE test_rocm.cpp)
endif()

target_link_libraries(iom_tests
    PRIVATE
        doctest::doctest
        libiom
)

add_test(NAME iom_tests COMMAND iom_tests)
