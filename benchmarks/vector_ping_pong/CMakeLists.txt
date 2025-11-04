cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(vector_ping_pong CXX)

include(FetchContent)
FetchContent_Declare(
    Kokkos
    URL      https://github.com/kokkos/kokkos/releases/download/4.7.01/kokkos-4.7.01.tar.gz
    URL_HASH SHA256=404cf33e76159e83b8b4ad5d86f6899d442b5da4624820ab457412116cdcd201
)
FetchContent_MakeAvailable(Kokkos)

add_executable(vector_ping_pong vector_ping_pong-kokkos.cpp)
target_link_libraries(vector_ping_pong PRIVATE Kokkos::kokkos)
