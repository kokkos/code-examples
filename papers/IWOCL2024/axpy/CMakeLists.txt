cmake_minimum_required(VERSION 3.16)

project(AXPBY)

find_package(Kokkos REQUIRED)

add_executable(axpby.exe main.cpp)
target_include_directories(axpby.exe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(axpby.exe PRIVATE Kokkos::kokkos)
