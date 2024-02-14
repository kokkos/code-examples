cmake_minimum_required(VERSION 3.16)

project(DOT)

find_package(Kokkos REQUIRED)

add_executable(dot.exe main.cpp)
target_include_directories(dot.exe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(dot.exe PRIVATE Kokkos::kokkos)
