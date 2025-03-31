cmake_minimum_required(VERSION 3.16)
project(AXPBY-DOT)
include(../common.cmake)

add_executable(axpby-dot main.cpp)
target_link_libraries(axpby-dot Kokkos::kokkos)


