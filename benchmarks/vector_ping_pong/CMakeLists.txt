cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(vector_ping_pong CXX)

include(FetchContent)
FetchContent_Declare(
    Kokkos
    URL      https://github.com/kokkos/kokkos/releases/download/5.0.0/kokkos-5.0.0.tar.gz
    URL_HASH SHA256=c45f3e19c3eb71fc8b7210cb04cac658015fc1839e7cc0571f7406588ff9bcef
)
FetchContent_MakeAvailable(Kokkos)

add_executable(vector_ping_pong vector_ping_pong-kokkos.cpp)
target_link_libraries(vector_ping_pong PRIVATE Kokkos::kokkos)
