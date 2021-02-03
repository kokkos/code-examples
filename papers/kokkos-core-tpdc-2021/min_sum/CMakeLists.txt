cmake_minimum_required(VERSION 3.16)
project(min_sum CXX)
find_package(Kokkos REQUIRED PATHS ${KOKKOS_DIR})

add_executable(min_sum min_sum.cpp)
target_link_libraries(min_sum PRIVATE Kokkos::kokkoscore)
