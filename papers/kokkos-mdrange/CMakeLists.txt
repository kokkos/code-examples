cmake_minimum_required(VERSION 3.28.2)
include(FetchContent)

project(mdrange-benchmarks LANGUAGES CXX)

set(CMAKE_COLOR_DIAGNOSTICS ON)

set(VERSION "502" CACHE STRING "Kokkos versions to build against")

if(VERSION STREQUAL "502")
  set(VERSION_COMMIT_SHA "dfceb77277e23361cccfbcf72c0468fd8475153e")
elseif(VERSION STREQUAL "device_iterate")
  set(VERSION_COMMIT_SHA "7a78e92a38f4f40f6dc84766d8b5acd14452e77b")  # Fix using non device-copyable functors in MDRangePolicy
elseif(VERSION STREQUAL "tile")
  set(VERSION_COMMIT_SHA "c6ab03f33c679e180f804dc706d53822e353c6e9")
elseif(VERSION STREQUAL "no_stride")
  set(VERSION_COMMIT_SHA "b77b924f9325ff17261f2fe05cca0423cd0b1fa4")
else()
  message(FATAL_ERROR "Unsupported version: ${VERSION}")
endif()

set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_INSTALL ON CACHE BOOL "" FORCE)
set(BENCHMARK_DOWNLOAD_DEPENDENCIES ON CACHE BOOL "" FORCE)
set(HAVE_STD_REGEX OFF CACHE BOOL "" FORCE)

message(STATUS "CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}")
message(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

find_package(benchmark QUIET)
if(benchmark_FOUND)
  message(STATUS "Found pre-built Google Benchmark")
else()
  set(BENCHMARK_COMMON_SOURCE_DIR ${CMAKE_SOURCE_DIR}/benchmark-1.8.4/)
  if(EXISTS ${BENCHMARK_COMMON_SOURCE_DIR})
    message(STATUS "Google Benchmark found in ${BENCHMARK_COMMON_SOURCE_DIR}, building from source")
    add_subdirectory(${BENCHMARK_COMMON_SOURCE_DIR})
  else()
    message(STATUS "Google Benchmark not found. Fetching from GitHub...")
    FetchContent_Declare(
      benchmark
      GIT_REPOSITORY https://github.com/google/benchmark.git
      GIT_TAG        v1.8.4
      SOURCE_DIR ${BENCHMARK_COMMON_SOURCE_DIR}
    )
    FetchContent_MakeAvailable(benchmark)
  endif()
endif()

set(KOKKOS_SOURCE_DIR ${CMAKE_SOURCE_DIR}/kokkos_${VERSION}/)
if(EXISTS ${KOKKOS_SOURCE_DIR})
  message(STATUS "Kokkos ${VERSION} found in ${KOKKOS_SOURCE_DIR}")
  add_subdirectory(${KOKKOS_SOURCE_DIR} ${CMAKE_BINARY_DIR}/kokkos-build)
else()
  message(STATUS "Kokkos ${VERSION} not found. Fetching from GitHub...")
  FetchContent_Declare(
    Kokkos
    GIT_REPOSITORY https://github.com/kokkos/kokkos.git
    GIT_TAG        ${VERSION_COMMIT_SHA}
    SOURCE_DIR     ${KOKKOS_SOURCE_DIR}
  )
  FetchContent_MakeAvailable(Kokkos)
endif()

message(STATUS "Git log:\n${GIT_LOG}")
message(STATUS "Modified files:\n${MODIFIED_FILES}")

function(ADD_TARGET_LINK_KOKKOS)
  set(options)
  set(oneValueArgs TARGET SRC)
  set(multiValueArgs LINKS FLAGS)

  cmake_parse_arguments(PARSE_ARGV 0 FN_ARGS "${options}"  "${oneValueArgs}" "${multiValueArgs}")

  add_executable(${FN_ARGS_TARGET} ${FN_ARGS_SRC})
  target_compile_options(${FN_ARGS_TARGET} PRIVATE ${FN_ARGS_FLAGS})
  target_link_libraries(${FN_ARGS_TARGET} ${FN_ARGS_LINKS})
endfunction()

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  set(FLAGS_WARNINGS "/W4")
else()
  set(FLAGS_WARNINGS "-Wall")
endif()

add_target_link_kokkos(TARGET "bench_mdrange_stencil" SRC "src/bench_mdrange_stencil.cpp" FLAGS ${FLAGS_WARNINGS} LINKS "Kokkos::kokkos" "benchmark::benchmark")
add_target_link_kokkos(TARGET "bench_mdrange_stream" SRC "src/bench_mdrange_stream.cpp" FLAGS ${FLAGS_WARNINGS} LINKS "Kokkos::kokkos" "benchmark::benchmark")
add_target_link_kokkos(TARGET "bench_mdrange_gemm" SRC "src/bench_mdrange_gemm.cpp" FLAGS ${FLAGS_WARNINGS} LINKS "Kokkos::kokkos" "benchmark::benchmark")
add_target_link_kokkos(TARGET "bench_teamPolicy_mdrangePolicy" SRC "src/bench_mdrange_teamMD.cpp" FLAGS ${FLAGS_WARNINGS} LINKS "Kokkos::kokkos" "benchmark::benchmark")

