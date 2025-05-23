cmake_minimum_required(VERSION 3.16)
project(test-3d)
include(../common.cmake)

add_executable(redn redn.cpp)
target_link_libraries(redn Kokkos::kokkos)


