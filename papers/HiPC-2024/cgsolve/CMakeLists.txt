cmake_minimum_required(VERSION 3.16)
project(CGSolve)
include(../common.cmake)

add_executable(cgsolve main.cpp)
target_link_libraries(cgsolve Kokkos::kokkos)


