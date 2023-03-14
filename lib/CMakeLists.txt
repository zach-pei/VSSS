cmake_minimum_required(VERSION 2.8)

add_library(vine_lib vine.cpp tools.cpp)

find_package(OpenCV REQUIRED)   
target_link_libraries(vine_lib ${OpenCV_LIBS})
#set (CMAKE_CXX_FLAGS “${CMAKE_CXX_FLAGS} -g”)