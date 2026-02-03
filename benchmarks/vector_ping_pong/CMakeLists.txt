cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(vector_ping_pong CXX)

include(FetchContent)
FetchContent_Declare(
    Kokkos
    URL      https://github.com/kokkos/kokkos/releases/download/5.0.1/kokkos-5.0.1.tar.gz
    URL_HASH SHA256=cf7d8515ca993229929be9f051aecd8f93cde325adac8a4f82ed6848adace218
)
FetchContent_MakeAvailable(Kokkos)

add_executable(vector_ping_pong vector_ping_pong-kokkos.cpp)
target_link_libraries(vector_ping_pong PRIVATE Kokkos::kokkos)
