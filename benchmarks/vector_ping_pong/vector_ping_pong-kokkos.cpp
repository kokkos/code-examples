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

template <typename ViewType, typename VectorIndex>
KOKKOS_INLINE_FUNCTION void workload(ViewType const& view,
                                     const VectorIndex index, int num_ops) {
  using ValueType = typename ViewType::value_type;

  ValueType a1      = view(index);
  const ValueType b = a1;

  for (int f = 0; f < num_ops; f++) {
    a1 += b * a1;
  }

  view(index) = a1;
}

template <typename MemorySpacePing, typename MemorySpacePong,
          typename ExecutionSpacePing, typename ExecutionSpacePong,
          typename ExecutionSpaceFirstTouchPing,
          typename ExecutionSpaceFirstTouchPong, bool needs_deep_copy,
          typename VectorValue, typename VectorIndex>
std::tuple<int, double> run_benchmark(VectorValue* ping_data,
                                      VectorValue* pong_data, VectorIndex size,
                                      int warmup_runs, int num_pingpongs,
                                      VectorIndex stride, int num_pings,
                                      int num_pongs, int scratch_size,
                                      int num_teams, int num_repetions_per_team,
                                      int num_ops) {
  if (size % num_teams != 0) {
    Kokkos::abort("Size of array must be a multiple of num_teams \n");
  }
  auto team_size = size / num_teams;

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
          pong_view(idx) = VectorValue{1.0};
        });
    Kokkos::fence();
  }

  Kokkos::Timer timer;
  for (auto i = 0; i < num_pingpongs; ++i) {
    if constexpr (needs_deep_copy)
      Kokkos::deep_copy(ping_view, pong_view);
    else
      Kokkos::fence();
    for (auto j = 0; j < num_pings; ++j)
      Kokkos::parallel_for(
          "ping",
          Kokkos::TeamPolicy(ExecutionSpacePing(), num_teams, team_size)
              .set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
          KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
            const VectorIndex n = team.league_rank() * (team.team_size());
            for (int r = 0; r < num_repetions_per_team; r++) {
              Kokkos::parallel_for(
                  Kokkos::TeamThreadRange(team, n, n + team.team_size()),
                  [&](const VectorIndex& i) {
                    workload(ping_view, i, num_ops);
                  });
            }
          });
    if constexpr (needs_deep_copy)
      Kokkos::deep_copy(pong_view, ping_view);
    else
      Kokkos::fence();
    for (auto j = 0; j < num_pings; ++j)
      Kokkos::parallel_for(
          "pong",
          Kokkos::TeamPolicy(ExecutionSpacePing(), num_teams, team_size)
              .set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
          KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
            const VectorIndex n = team.league_rank() * (team.team_size());
            for (int r = 0; r < num_repetions_per_team; r++) {
              Kokkos::parallel_for(
                  Kokkos::TeamThreadRange(team, n, n + team.team_size()),
                  [&](const VectorIndex& i) {
                    workload(pong_view, i, num_ops);
                  });
            }
          });
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
        error += (ping_view(i) == static_cast<VectorValue>(num_pingpongs) *
                                      (num_pings + num_pongs))
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
                     IndexType stride, AllocatorPing, AllocatorPong, int pings,
                     int pongs, int scratch_size, int num_teams,
                     int num_repetions_per_team, int num_ops) {
  ValueType* vec_ping =
      AllocatorPing::template allocate<ValueType>(size * stride);
  ValueType* vec_pong =
      AllocatorPong::template allocate<ValueType>(size * stride);

  auto rc = run_benchmark<Kokkos::SharedSpace, Kokkos::SharedSpace,
                          ExecutionSpacePing, ExecutionSpacePong,
                          ExecutionSpaceFirstTouchPing,
                          ExecutionSpaceFirstTouchPong, true>(
      vec_ping, vec_pong, size, warmups, pingpongs, stride, pings, pongs,
      scratch_size, num_teams, num_repetions_per_team, num_ops);

  AllocatorPing::deallocate(vec_ping);
  AllocatorPong::deallocate(vec_pong);

  return rc;
}

template <typename ValueType, typename ExecutionSpacePing,
          typename ExecutionSpacePong, typename ExecutionSpaceFirstTouchPing,
          typename ExecutionSpaceFirstTouchPong, typename AllocatorPingPong,
          typename IndexType>
auto benchmark_views(IndexType size, int warmups, int pingpongs,
                     IndexType stride, AllocatorPingPong, NONE, int pings,
                     int pongs, int scratch_size, int num_teams,
                     int num_repetions_per_team, int num_ops) {
  ValueType* vec_ping_pong =
      AllocatorPingPong::template allocate<ValueType>(size * stride);

  auto rc = run_benchmark<Kokkos::SharedSpace, Kokkos::SharedSpace,
                          ExecutionSpacePing, ExecutionSpacePong,
                          ExecutionSpaceFirstTouchPing,
                          ExecutionSpaceFirstTouchPong, false>(
      vec_ping_pong, vec_ping_pong, size, warmups, pingpongs, stride, pings,
      pongs, scratch_size, num_teams, num_repetions_per_team, num_ops);

  AllocatorPingPong::deallocate(vec_ping_pong);

  return rc;
}

template <typename ValueType, typename IndexType, typename AllocatorPing,
          typename AllocatorPong = NONE>
void benchmark_and_print(std::ostream& out, unsigned const rep,
                         IndexType array_size, unsigned warmups,
                         unsigned pingpongs, IndexType stride,
                         AllocatorPing Aping, AllocatorPong Apong, int pings,
                         int pongs, int scratch_size, int num_teams,
                         int num_repetions_per_team, int num_ops) {
  auto [rc, timing] = benchmark_views<ValueType, Kokkos::DefaultExecutionSpace,
                                      Kokkos::DefaultHostExecutionSpace,
                                      Kokkos::DefaultExecutionSpace,
                                      Kokkos::DefaultHostExecutionSpace>(
      array_size, warmups, pingpongs, stride, Aping, Apong, pings, pongs,
      scratch_size, num_teams, num_repetions_per_team, num_ops);
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
      << typeid(AllocatorPong()).name() << " , " << pings << " , " << pongs
      << "\n";
}

int main(int argc, char* argv[]) {  // NOLINT(bugprone-exception-escape)
  Kokkos::initialize(argc, argv);
  {
    using ValueType = double;
    using IndexType = std::size_t;

    if (argc < 9)
      printf(
          "Arguments: prefix mode repetitions array_size "
          "warmup_runs ping_pongs stride pings pongs/n");

    const std::string prefix(argv[1]);
    const std::string mode(argv[2]);
    int repetitions      = std::stoi(argv[3]);
    IndexType array_size = std::stoi(argv[4]);
    int warmup_runs      = std::stoi(argv[5]);
    int ping_pongs       = std::stoi(argv[6]);
    IndexType stride     = std::stoi(argv[7]);
    int pings            = std::stoi(argv[8]);
    int pongs            = std::stoi(argv[9]);

    std::ofstream outfile;
    outfile.open(prefix + "_" + mode + "_" + argv[3] + "_" + argv[4] + "_" +
                     argv[5] + "_" + argv[6] + "_" + argv[7] + "_" + argv[8] +
                     "_" + argv[9] + ".csv",
                 std::ios::out);

    Kokkos::print_configuration(outfile);

    outfile << "# repetition, arraysize, warmups, pingpongs, stride, "
               "bandwidth, time, "
               "allocatorPing, "
               "allocatorPong, "
               "pings, "
               "pongs "
            << std::endl;
    // TODO get these values from the input and write to output. Also adapt the
    // run scripts to do that.
    int scratch_size, num_teams, num_repetions_per_team, num_ops;
    // TODO
    // revisit check of output ... if runs are repeated the check we currently
    // run should give the wrong answer to compare against.

    for (int rep = 0; rep <= repetitions; ++rep) {
      // TWO VIEWS
      // MANAGED
      if (mode == "managed-managed")
        benchmark_and_print<ValueType>(
            outfile, rep, array_size, warmup_runs, ping_pongs, stride,
            ManagedMalloc(), ManagedMalloc(), pings, pongs, scratch_size,
            num_teams, num_repetions_per_team, num_ops);
      else if (mode == "managed-new")
        benchmark_and_print<ValueType>(
            outfile, rep, array_size, warmup_runs, ping_pongs, stride,
            ManagedMalloc(), StdNew(), pings, pongs, scratch_size, num_teams,
            num_repetions_per_team, num_ops);
      else if (mode == "managed-malloc")
        benchmark_and_print<ValueType>(
            outfile, rep, array_size, warmup_runs, ping_pongs, stride,
            ManagedMalloc(), StdMalloc(), pings, pongs, scratch_size, num_teams,
            num_repetions_per_team, num_ops);
      else if (mode == "managed-hostpinned")
        benchmark_and_print<ValueType>(
            outfile, rep, array_size, warmup_runs, ping_pongs, stride,
            ManagedMalloc(), HostPinnedMalloc(), pings, pongs, scratch_size,
            num_teams, num_repetions_per_team, num_ops);
      else if (mode == "managed-device")
        benchmark_and_print<ValueType>(
            outfile, rep, array_size, warmup_runs, ping_pongs, stride,
            ManagedMalloc(), DeviceMalloc(), pings, pongs, scratch_size,
            num_teams, num_repetions_per_team, num_ops);

      // DEVICE
      else if (mode == "device-managed")
        benchmark_and_print<ValueType>(
            outfile, rep, array_size, warmup_runs, ping_pongs, stride,
            DeviceMalloc(), ManagedMalloc(), pings, pongs, scratch_size,
            num_teams, num_repetions_per_team, num_ops);
      else if (mode == "device-new")
        benchmark_and_print<ValueType>(
            outfile, rep, array_size, warmup_runs, ping_pongs, stride,
            DeviceMalloc(), StdNew(), pings, pongs, scratch_size, num_teams,
            num_repetions_per_team, num_ops);
      else if (mode == "device-malloc")
        benchmark_and_print<ValueType>(
            outfile, rep, array_size, warmup_runs, ping_pongs, stride,
            DeviceMalloc(), StdMalloc(), pings, pongs, scratch_size, num_teams,
            num_repetions_per_team, num_ops);
      else if (mode == "device-hostpinned")
        benchmark_and_print<ValueType>(
            outfile, rep, array_size, warmup_runs, ping_pongs, stride,
            DeviceMalloc(), HostPinnedMalloc(), pings, pongs, scratch_size,
            num_teams, num_repetions_per_team, num_ops);
      else if (mode == "device-device")
        benchmark_and_print<ValueType>(
            outfile, rep, array_size, warmup_runs, ping_pongs, stride,
            DeviceMalloc(), DeviceMalloc(), pings, pongs, scratch_size,
            num_teams, num_repetions_per_team, num_ops);

      // HostPinned
      else if (mode == "hostpinned-managed")
        benchmark_and_print<ValueType>(
            outfile, rep, array_size, warmup_runs, ping_pongs, stride,
            DeviceMalloc(), ManagedMalloc(), pings, pongs, scratch_size,
            num_teams, num_repetions_per_team, num_ops);
      else if (mode == "hostpinned-new")
        benchmark_and_print<ValueType>(
            outfile, rep, array_size, warmup_runs, ping_pongs, stride,
            DeviceMalloc(), StdNew(), pings, pongs, scratch_size, num_teams,
            num_repetions_per_team, num_ops);
      else if (mode == "hostpinned-malloc")
        benchmark_and_print<ValueType>(
            outfile, rep, array_size, warmup_runs, ping_pongs, stride,
            DeviceMalloc(), StdMalloc(), pings, pongs, scratch_size, num_teams,
            num_repetions_per_team, num_ops);
      else if (mode == "hostpinned-hostpinned")
        benchmark_and_print<ValueType>(
            outfile, rep, array_size, warmup_runs, ping_pongs, stride,
            DeviceMalloc(), HostPinnedMalloc(), pings, pongs, scratch_size,
            num_teams, num_repetions_per_team, num_ops);
      else if (mode == "hostpinned-device")
        benchmark_and_print<ValueType>(
            outfile, rep, array_size, warmup_runs, ping_pongs, stride,
            DeviceMalloc(), DeviceMalloc(), pings, pongs, scratch_size,
            num_teams, num_repetions_per_team, num_ops);

      // NEW
      else if (mode == "new-managed")
        benchmark_and_print<ValueType>(
            outfile, rep, array_size, warmup_runs, ping_pongs, stride, StdNew(),
            ManagedMalloc(), pings, pongs, scratch_size, num_teams,
            num_repetions_per_team, num_ops);
      else if (mode == "new-new")
        benchmark_and_print<ValueType>(outfile, rep, array_size, warmup_runs,
                                       ping_pongs, stride, StdNew(), StdNew(),
                                       pings, pongs, scratch_size, num_teams,
                                       num_repetions_per_team, num_ops);
      else if (mode == "new-malloc")
        benchmark_and_print<ValueType>(
            outfile, rep, array_size, warmup_runs, ping_pongs, stride, StdNew(),
            StdMalloc(), pings, pongs, scratch_size, num_teams,
            num_repetions_per_team, num_ops);
      else if (mode == "new-hostpinned")
        benchmark_and_print<ValueType>(
            outfile, rep, array_size, warmup_runs, ping_pongs, stride, StdNew(),
            HostPinnedMalloc(), pings, pongs, scratch_size, num_teams,
            num_repetions_per_team, num_ops);
      else if (mode == "new-device")
        benchmark_and_print<ValueType>(
            outfile, rep, array_size, warmup_runs, ping_pongs, stride, StdNew(),
            DeviceMalloc(), pings, pongs, scratch_size, num_teams,
            num_repetions_per_team, num_ops);

      // ONE VIEW
      // MANAGED
      else if (mode == "managed-none")
        benchmark_and_print<ValueType>(
            outfile, rep, array_size, warmup_runs, ping_pongs, stride,
            ManagedMalloc(), NONE(), pings, pongs, scratch_size, num_teams,
            num_repetions_per_team, num_ops);
      else if (mode == "hostpinned-none")
        benchmark_and_print<ValueType>(
            outfile, rep, array_size, warmup_runs, ping_pongs, stride,
            ManagedMalloc(), NONE(), pings, pongs, scratch_size, num_teams,
            num_repetions_per_team, num_ops);
      else if (mode == "device-none")
        benchmark_and_print<ValueType>(
            outfile, rep, array_size, warmup_runs, ping_pongs, stride,
            DeviceMalloc(), NONE(), pings, pongs, scratch_size, num_teams,
            num_repetions_per_team, num_ops);
      else if (mode == "malloc-none")
        benchmark_and_print<ValueType>(outfile, rep, array_size, warmup_runs,
                                       ping_pongs, stride, StdMalloc(), NONE(),
                                       pings, pongs, scratch_size, num_teams,
                                       num_repetions_per_team, num_ops);
      else if (mode == "new-none")
        benchmark_and_print<ValueType>(outfile, rep, array_size, warmup_runs,
                                       ping_pongs, stride, StdNew(), NONE(),
                                       pings, pongs, scratch_size, num_teams,
                                       num_repetions_per_team, num_ops);
    }
    outfile.close();
  }
  Kokkos::finalize();

  return 0;
}
