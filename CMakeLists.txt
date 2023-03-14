cmake_minimum_required(VERSION 2.8)

project(vine_spread)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)


add_subdirectory(lib)
add_subdirectory(main)

set (CMAKE_CXX_FLAGS “${CMAKE_CXX_FLAGS} -g”)
