cmake_minimum_required(VERSION 3.16)

project(CGSolve)

find_package(Kokkos REQUIRED)

add_executable(cg_solve.exe main.cpp)

target_include_directories(cg_solve.exe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cg_solve.exe PRIVATE Kokkos::kokkos)

if(USE_MKL)
  find_package(KokkosKernels REQUIRED)
  find_package(MKL)
  target_compile_definitions(cg_solve.exe PRIVATE USE_MKL)
  target_link_libraries(cg_solve.exe PRIVATE Kokkos::kokkoskernels)
endif()
