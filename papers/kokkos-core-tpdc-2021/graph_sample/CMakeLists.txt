# Kokkos minimally requires 3.10 right now,
# but your project can set it higher
cmake_minimum_required(VERSION 3.16)

# Projects can safely mix languages - must have C++ support
# Kokkos flags will only apply to C++ files
project(graph_sample CXX)
# Look for an installed Kokkos
find_package(Kokkos REQUIRED)

add_executable(graph_sample graph_sample.cpp)

# This is the only thing required to set up compiler/linker flags
target_link_libraries(graph_sample Kokkos::kokkos)

