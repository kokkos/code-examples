//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#include "Kokkos_Core.hpp"
#include "Kokkos_Macros.hpp"

#include <iostream>
#include <fstream>

#include <sys/time.h>

template <typename MemorySpacePing, typename MemorySpacePong,
          typename ExecutionSpacePing, typename ExecutionSpacePong,
          typename ExecutionSpaceFirstTouchPing,
          typename ExecutionSpaceFirstTouchPong, bool needs_deep_copy,
          typename VectorValue, typename VectorIndex>
std::tuple<int, double> run_benchmark(VectorValue* ping_data,
                                      VectorValue* pong_data, VectorIndex size,
                                      int warmup_runs, int num_pingpongs,
                                      VectorIndex stride) {
  auto warmup_view =
      Kokkos::View<VectorValue*, MemorySpacePing>{"warmup", size};

  Kokkos::LayoutStride layout_stride(size, stride);
  auto ping_view =
      Kokkos::View<VectorValue*, Kokkos::LayoutStride, MemorySpacePing>{
          ping_data, layout_stride};
  auto pong_view =
      Kokkos::View<VectorValue*, Kokkos::LayoutStride, MemorySpacePong>{
          pong_data, layout_stride};

  // do warmup with another view so we don't mess up the placement
  for (auto i = 0; i < warmup_runs; ++i) {
    Kokkos::parallel_for(
        "warmup inc", Kokkos::RangePolicy(ExecutionSpacePing(), 0, size),
        KOKKOS_LAMBDA(const VectorIndex idx) { ++warmup_view(idx); });
    Kokkos::parallel_for(
        "warmup dec", Kokkos::RangePolicy(ExecutionSpacePing(), 0, size),
        KOKKOS_LAMBDA(const VectorIndex idx) { --warmup_view(idx); });
  }
  Kokkos::fence();

  Kokkos::parallel_for(
      "first_touch_ping",
      Kokkos::RangePolicy(ExecutionSpaceFirstTouchPing(), 0, size),
      KOKKOS_LAMBDA(const VectorIndex idx) { ping_view(idx) = VectorValue{}; });
  Kokkos::fence();

  // if it needs a deepcopy there are two arrays, thus it also needs a pong
  // first touch
  if constexpr (needs_deep_copy) {
    Kokkos::parallel_for(
        "first_touch_pong",
        Kokkos::RangePolicy(ExecutionSpaceFirstTouchPong(), 0, size),
        KOKKOS_LAMBDA(const VectorIndex idx) {
          pong_view(idx) = VectorValue{};
        });
    Kokkos::fence();
  }

  Kokkos::Timer timer;
  for (auto i = 0; i < num_pingpongs; ++i) {
    if constexpr (needs_deep_copy)
      Kokkos::deep_copy(ping_view, pong_view);
    else
      Kokkos::fence();
    Kokkos::parallel_for(
        "ping", Kokkos::RangePolicy(ExecutionSpacePing(), 0, size),
        KOKKOS_LAMBDA(const VectorIndex idx) { ++ping_view(idx); });
    if constexpr (needs_deep_copy)
      Kokkos::deep_copy(pong_view, ping_view);
    else
      Kokkos::fence();
    Kokkos::parallel_for(
        "pong", Kokkos::RangePolicy(ExecutionSpacePong(), 0, size),
        KOKKOS_LAMBDA(const VectorIndex idx) { ++pong_view(idx); });
  }
  Kokkos::fence();
  auto totalTime = timer.seconds();

  // check for errors
  int error_count = 0;
  // since we ended on pong but want to check ping, we need to copy it again.
  Kokkos::deep_copy(ping_view, pong_view);
  Kokkos::parallel_reduce(
      "error_check", Kokkos::RangePolicy(ExecutionSpacePing(), 0, size),
      KOKKOS_LAMBDA(const VectorIndex i, int& error) {
        error += (ping_view(i) == static_cast<VectorValue>(num_pingpongs) * 2)
                     ? 0
                     : 1;
      },
      error_count);
  Kokkos::fence();
  return std::make_tuple(error_count, totalTime);
}

//// ALLOCATOR DEALLOCATOR
struct ManagedMalloc {
  template <typename T>
  static constexpr T* allocate(size_t size) {
    T* ptr;
#ifdef KOKKOS_ENABLE_CUDA
    cudaMallocManaged(&ptr, size * sizeof(T));
#elif defined KOKKOS_ENABLE_HIP
    hipMallocManaged(&ptr, size * sizeof(T));
#endif
    return ptr;
  }

  template <typename T>
  static constexpr void deallocate(T* ptr) {
#ifdef KOKKOS_ENABLE_CUDA
    cudaFree(ptr);
#elif defined KOKKOS_ENABLE_HIP
    hipFree(ptr);
#endif
  }
};

struct HostPinnedMalloc {
  template <typename T>
  static constexpr T* allocate(size_t size) {
    T* ptr;
#ifdef KOKKOS_ENABLE_CUDA
    cudaMallocHost(&ptr, size * sizeof(T));
#elif defined KOKKOS_ENABLE_HIP
    hipHostMalloc(&ptr, size * sizeof(T));
#endif
    return ptr;
  }

  template <typename T>
  static constexpr void deallocate(T* ptr) {
#ifdef KOKKOS_ENABLE_CUDA
    cudaFree(ptr);
#elif defined KOKKOS_ENABLE_HIP
    hipFree(ptr);
#endif
  }
};

struct DeviceMalloc {
  template <typename T>
  static constexpr T* allocate(size_t size) {
    T* ptr;
#ifdef KOKKOS_ENABLE_CUDA
    cudaMalloc(&ptr, size * sizeof(T));
#elif defined KOKKOS_ENABLE_HIP
    hipMalloc(&ptr, size * sizeof(T));
#endif
    return ptr;
  }

  template <typename T>
  static constexpr void deallocate(T* ptr) {
#ifdef KOKKOS_ENABLE_CUDA
    cudaFree(ptr);
#elif defined KOKKOS_ENABLE_HIP
    hipFree(ptr);
#endif
  }
};

struct StdMalloc {
  template <typename T>
  static constexpr T* allocate(size_t size) {
    return static_cast<T*>(std::malloc(size * sizeof(T)));
  }

  template <typename T>
  static constexpr void deallocate(T* ptr) {
    std::free(ptr);
  }
};

struct StdNew {
  template <typename T>
  static constexpr T* allocate(size_t size) {
    return new T[size];
  }

  template <typename T>
  static constexpr void deallocate(T* ptr) {
    delete[] ptr;
  }
};

struct NONE {};

///////////////////////single and double array
template <typename ValueType, typename ExecutionSpacePing,
          typename ExecutionSpacePong, typename ExecutionSpaceFirstTouchPing,
          typename ExecutionSpaceFirstTouchPong, typename AllocatorPing,
          typename AllocatorPong, typename IndexType>
auto benchmark_views(IndexType size, int warmups, int pingpongs,
                     IndexType stride, AllocatorPing, AllocatorPong) {
  ValueType* vec_ping =
      AllocatorPing::template allocate<ValueType>(size * stride);
  ValueType* vec_pong =
      AllocatorPong::template allocate<ValueType>(size * stride);

  auto rc =
      run_benchmark<Kokkos::SharedSpace, Kokkos::SharedSpace,
                    ExecutionSpacePing, ExecutionSpacePong,
                    ExecutionSpaceFirstTouchPing, ExecutionSpaceFirstTouchPong,
                    true>(vec_ping, vec_pong, size, warmups, pingpongs, stride);

  AllocatorPing::deallocate(vec_ping);
  AllocatorPong::deallocate(vec_pong);

  return rc;
}

template <typename ValueType, typename ExecutionSpacePing,
          typename ExecutionSpacePong, typename ExecutionSpaceFirstTouchPing,
          typename ExecutionSpaceFirstTouchPong, typename AllocatorPingPong,
          typename IndexType>
auto benchmark_views(IndexType size, int warmups, int pingpongs,
                     IndexType stride, AllocatorPingPong, NONE) {
  ValueType* vec_ping_pong =
      AllocatorPingPong::template allocate<ValueType>(size * stride);

  auto rc = run_benchmark<Kokkos::SharedSpace, Kokkos::SharedSpace,
                          ExecutionSpacePing, ExecutionSpacePong,
                          ExecutionSpaceFirstTouchPing,
                          ExecutionSpaceFirstTouchPong, false>(
      vec_ping_pong, vec_ping_pong, size, warmups, pingpongs, stride);

  AllocatorPingPong::deallocate(vec_ping_pong);

  return rc;
}

template <typename ValueType, typename IndexType, typename AllocatorPing,
          typename AllocatorPong = NONE>
void benchmark_and_print(std::ostream& out, unsigned const rep,
                         IndexType array_size, unsigned warmups,
                         unsigned pingpongs, IndexType stride,
                         AllocatorPing Aping, AllocatorPong Apong) {
  auto [rc, timing] = benchmark_views<ValueType, Kokkos::DefaultExecutionSpace,
                                      Kokkos::DefaultHostExecutionSpace,
                                      Kokkos::DefaultExecutionSpace,
                                      Kokkos::DefaultHostExecutionSpace>(
      array_size, warmups, pingpongs, stride, Aping, Apong);
  if (rc != 0) {
    std::cout << "WRONG RESULT in rep " << rep << " array_size " << array_size
              << " warmups " << warmups << " pingpongs " << pingpongs
              << " stride " << stride << ".  exiting!" << std::endl;
    std::exit(rc);
  }

  double bw = 1.0e-6 * 2.0 * pingpongs * array_size *
              (double)sizeof(ValueType) / timing;
  out << rep << " , " << array_size << " , " << warmups << " , " << pingpongs
      << " , " << stride << " , " << bw << " , " << timing << " , "
      << typeid(AllocatorPing()).name() << " , "
      << typeid(AllocatorPong()).name() << "\n";
}

int main(int argc, char* argv[]) {  // NOLINT(bugprone-exception-escape)
  Kokkos::initialize(argc, argv);
  {
    using ValueType = int;
    using IndexType = unsigned int;

    if (argc < 6)
      printf(
          "Arguments: mode repetitions array_size "
          "warmup_runs ping_pongs stride/n");

    const std::string mode(argv[1]);
    int repetitions      = std::stoi(argv[2]);
    IndexType array_size = std::stoi(argv[3]);
    int warmup_runs      = std::stoi(argv[4]);
    int ping_pongs       = std::stoi(argv[5]);
    IndexType stride     = std::stoi(argv[6]);

    std::ofstream outfile;
    outfile.open(mode + "_" + argv[3] + "_" + argv[4] + "_" + argv[5] + "_" +
                     argv[6] + ".csv",
                 std::ios::out);

    Kokkos::print_configuration(outfile);

    outfile << "# repetition, arraysize, warmups, pingpongs, stride, "
               "bandwidth, time, "
               "allocatorPing, "
               "allocatorPong"
            << std::endl;

    for (int rep = 0; rep <= repetitions; ++rep) {
      // TWO VIEWS
      // MANAGED
      if (mode == "managed-managed")
        benchmark_and_print<ValueType>(outfile, rep, array_size, warmup_runs,
                                       ping_pongs, stride, ManagedMalloc(),
                                       ManagedMalloc());
      else if (mode == "managed-new")
        benchmark_and_print<ValueType>(outfile, rep, array_size, warmup_runs,
                                       ping_pongs, stride, ManagedMalloc(),
                                       StdNew());
      else if (mode == "managed-malloc")
        benchmark_and_print<ValueType>(outfile, rep, array_size, warmup_runs,
                                       ping_pongs, stride, ManagedMalloc(),
                                       StdMalloc());
      else if (mode == "managed-hostpinned")
        benchmark_and_print<ValueType>(outfile, rep, array_size, warmup_runs,
                                       ping_pongs, stride, ManagedMalloc(),
                                       HostPinnedMalloc());
      else if (mode == "managed-device")
        benchmark_and_print<ValueType>(outfile, rep, array_size, warmup_runs,
                                       ping_pongs, stride, ManagedMalloc(),
                                       DeviceMalloc());

      // DEVICE
      else if (mode == "device-managed")
        benchmark_and_print<ValueType>(outfile, rep, array_size, warmup_runs,
                                       ping_pongs, stride, DeviceMalloc(),
                                       ManagedMalloc());
      else if (mode == "device-new")
        benchmark_and_print<ValueType>(outfile, rep, array_size, warmup_runs,
                                       ping_pongs, stride, DeviceMalloc(),
                                       StdNew());
      else if (mode == "device-malloc")
        benchmark_and_print<ValueType>(outfile, rep, array_size, warmup_runs,
                                       ping_pongs, stride, DeviceMalloc(),
                                       StdMalloc());
      else if (mode == "device-hostpinned")
        benchmark_and_print<ValueType>(outfile, rep, array_size, warmup_runs,
                                       ping_pongs, stride, DeviceMalloc(),
                                       HostPinnedMalloc());
      else if (mode == "device-device")
        benchmark_and_print<ValueType>(outfile, rep, array_size, warmup_runs,
                                       ping_pongs, stride, DeviceMalloc(),
                                       DeviceMalloc());

      // HostPinned
      else if (mode == "hostpinned-managed")
        benchmark_and_print<ValueType>(outfile, rep, array_size, warmup_runs,
                                       ping_pongs, stride, DeviceMalloc(),
                                       ManagedMalloc());
      else if (mode == "hostpinned-new")
        benchmark_and_print<ValueType>(outfile, rep, array_size, warmup_runs,
                                       ping_pongs, stride, DeviceMalloc(),
                                       StdNew());
      else if (mode == "hostpinned-malloc")
        benchmark_and_print<ValueType>(outfile, rep, array_size, warmup_runs,
                                       ping_pongs, stride, DeviceMalloc(),
                                       StdMalloc());
      else if (mode == "hostpinned-hostpinned")
        benchmark_and_print<ValueType>(outfile, rep, array_size, warmup_runs,
                                       ping_pongs, stride, DeviceMalloc(),
                                       HostPinnedMalloc());
      else if (mode == "hostpinned-device")
        benchmark_and_print<ValueType>(outfile, rep, array_size, warmup_runs,
                                       ping_pongs, stride, DeviceMalloc(),
                                       DeviceMalloc());

      // NEW
      else if (mode == "new-managed")
        benchmark_and_print<ValueType>(outfile, rep, array_size, warmup_runs,
                                       ping_pongs, stride, StdNew(),
                                       ManagedMalloc());
      else if (mode == "new-new")
        benchmark_and_print<ValueType>(outfile, rep, array_size, warmup_runs,
                                       ping_pongs, stride, StdNew(), StdNew());
      else if (mode == "new-malloc")
        benchmark_and_print<ValueType>(outfile, rep, array_size, warmup_runs,
                                       ping_pongs, stride, StdNew(),
                                       StdMalloc());
      else if (mode == "new-hostpinned")
        benchmark_and_print<ValueType>(outfile, rep, array_size, warmup_runs,
                                       ping_pongs, stride, StdNew(),
                                       HostPinnedMalloc());
      else if (mode == "new-device")
        benchmark_and_print<ValueType>(outfile, rep, array_size, warmup_runs,
                                       ping_pongs, stride, StdNew(),
                                       DeviceMalloc());

      // ONE VIEW
      // MANAGED
      else if (mode == "managed-none")
        benchmark_and_print<ValueType>(outfile, rep, array_size, warmup_runs,
                                       ping_pongs, stride, ManagedMalloc(),
                                       NONE());
      else if (mode == "hostpinned-none")
        benchmark_and_print<ValueType>(outfile, rep, array_size, warmup_runs,
                                       ping_pongs, stride, ManagedMalloc(),
                                       NONE());
      else if (mode == "device-none")
        benchmark_and_print<ValueType>(outfile, rep, array_size, warmup_runs,
                                       ping_pongs, stride, DeviceMalloc(),
                                       NONE());
      else if (mode == "malloc-none")
        benchmark_and_print<ValueType>(outfile, rep, array_size, warmup_runs,
                                       ping_pongs, stride, StdMalloc(), NONE());
      else if (mode == "new-none")
        benchmark_and_print<ValueType>(outfile, rep, array_size, warmup_runs,
                                       ping_pongs, stride, StdNew(), NONE());
    }
    outfile.close();
  }
  Kokkos::finalize();

  return 0;
}
