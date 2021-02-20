cmake_minimum_required(VERSION 3.16)
project(scatter_view CXX)
find_package(Kokkos REQUIRED PATHS ${KOKKOS_DIR})

add_executable(scatter_view scatter_view.cpp)
target_link_libraries(scatter_view PRIVATE Kokkos::kokkoscore)
