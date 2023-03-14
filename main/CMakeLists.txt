include_directories(../lib/)

find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS system filesystem program_options REQUIRED)

add_executable(main main.cpp)
target_link_libraries(main ${Boost_LIBRARIES} ${OpenCV_LIBS} vine_lib)