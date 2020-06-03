set(FILE_TEST
    case.cc
    test_main.cc
    test_quadeigs.cc)

add_executable(unit_tests
    ${FILE_TEST})
 
target_link_libraries(unit_tests
    PRIVATE
    fmt-header-only)