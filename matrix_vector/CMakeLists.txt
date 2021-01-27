cmake_minimum_required(VERSION 3.16)
project(matrix_vector CXX)
find_package(Kokkos REQUIRED PATHS ${KOKKOS_DIR})

add_executable(matrix_vector matrix_vector.cpp)
target_link_libraries(matrix_vector PRIVATE Kokkos::kokkoscore)
