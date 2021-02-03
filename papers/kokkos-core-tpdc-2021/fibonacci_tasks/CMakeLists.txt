cmake_minimum_required(VERSION 3.16)
project(fibonacci_tasks CXX)
find_package(Kokkos REQUIRED PATHS ${KOKKOS_DIR})

add_executable(fibonacci_tasks fibonacci_tasks.cpp)
target_link_libraries(fibonacci_tasks PRIVATE Kokkos::kokkoscore)
