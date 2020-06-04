set(FILE_TEST
    case.cc
    test_main.cc
    test_quadeigs.cc)

aux_source_directory(${CMAKE_SOURCE_DIR}/src FILE_SRC)

add_executable(unit_tests
    ${FILE_TEST}
    ${FILE_SRC}
    )
 
target_link_libraries(unit_tests
    PRIVATE
    fmt-header-only)