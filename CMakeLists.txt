cmake_minimum_required(VERSION 3.0.0)
project(QuadEigsSOAR VERSION 0.1.0)

set(CMAKE_CXX_FLAGS "-Wall -g -fPIC -O3 -march=native")
# set(CMAKE_CXX_FLAGS "-Wall -g")

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
include_directories(${EIGEN3_INCLUDE_DIR})

include(CTest)
enable_testing()

include_directories(${CMAKE_SOURCE_DIR}/include)

# fmt
add_subdirectory(${CMAKE_SOURCE_DIR}/external/fmt EXCLUDE_FROM_ALL)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

add_subdirectory(test)
add_subdirectory(src)

enable_testing()
add_test(NAME test_all COMMAND unit_tests)
