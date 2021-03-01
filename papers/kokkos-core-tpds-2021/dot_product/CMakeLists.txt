cmake_minimum_required(VERSION 3.16)
project(dot_product CXX)
find_package(Kokkos REQUIRED PATHS ${KOKKOS_DIR})

add_executable(dot_product dot_product.cpp)
target_link_libraries(dot_product PRIVATE Kokkos::kokkoscore)
