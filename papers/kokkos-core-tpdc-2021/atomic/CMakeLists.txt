cmake_minimum_required(VERSION 3.16)
project(atomic CXX)
find_package(Kokkos REQUIRED PATHS ${KOKKOS_DIR})

add_executable(atomic atomic.cpp)
target_link_libraries(atomic PRIVATE Kokkos::kokkoscore)
