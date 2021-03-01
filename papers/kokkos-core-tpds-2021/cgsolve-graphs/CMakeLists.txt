cmake_minimum_required(VERSION 3.16)
project(cgsolve CXX)
find_package(Kokkos REQUIRED PATHS ${KOKKOS_DIR})
find_package(KokkosKernels REQUIRED)

add_executable(cgsolve cgsolve.cpp)
target_link_libraries(cgsolve PRIVATE Kokkos::kokkoscore Kokkos::kokkoskernels)
