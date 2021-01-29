cmake_minimum_required(VERSION 3.16)
project(mdrange CXX)
find_package(Kokkos REQUIRED PATHS ${KOKKOS_DIR})

add_executable(mdrange mdrange.cpp)
target_link_libraries(mdrange PRIVATE Kokkos::kokkoscore)
