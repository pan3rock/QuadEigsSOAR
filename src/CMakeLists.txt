set(CMAKE_LIBRARY_OUTPUT_DIRECTORY
    ${CMAKE_SOURCE_DIR}/quadeigs)

set(FILE_SRC
    soar.cc
    quadeigs.cc
    )

add_library(soar
    ${FILE_SRC})

target_link_libraries(soar
    PRIVATE
    fmt-header-only
    )