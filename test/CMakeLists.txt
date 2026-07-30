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
