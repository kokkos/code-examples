// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <functional>
#include <iostream>
#include <numeric>
#include <limits>

#include <benchmark/benchmark.h>

#include <Kokkos_Core.hpp>

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  Kokkos::print_configuration(std::cout, true);
  benchmark::Initialize(&argc, argv);
  benchmark::SetDefaultTimeUnit(benchmark::kMillisecond);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  Kokkos::finalize();
  return 0;
}

namespace KokkosBenchmark {

// Mark the label as a figure of merit.
inline std::string benchmark_fom(const std::string &label) {
  return "FOM: " + label;
}

// Report throughput and amount of data processed for simple View operations
template <class ViewType>
void report_results(benchmark::State &state, ViewType view, int data_ratio,
                    double time) {
  // data processed in megabytes
  const double data_processed = static_cast<double>(data_ratio) * view.size() *
                                sizeof(typename ViewType::value_type) /
                                1'000'000;

  state.SetIterationTime(time);
  state.counters["MB"] = benchmark::Counter(data_processed);
  state.counters[KokkosBenchmark::benchmark_fom("GB/s")] = benchmark::Counter(
      data_processed / 1'000, benchmark::Counter::kIsIterationInvariantRate);
}

}  // namespace KokkosBenchmark

template <typename Layout>
struct LayoutToIterationPattern {};

template <>
struct LayoutToIterationPattern<Kokkos::LayoutRight> {
  static constexpr Kokkos::Iterate pattern = Kokkos::Iterate::Right;
};

template <>
struct LayoutToIterationPattern<Kokkos::LayoutLeft> {
  static constexpr Kokkos::Iterate pattern = Kokkos::Iterate::Left;
};

template <int Rank, typename ScalarType, typename Layout>
struct ViewTypeRank {
  using type = void;
};

template <typename ScalarType, typename Layout>
struct ViewTypeRank<1, ScalarType, Layout> {
  using type = Kokkos::View<ScalarType *, Layout>;
};

template <typename ScalarType, typename Layout>
struct ViewTypeRank<2, ScalarType, Layout> {
  using type = Kokkos::View<ScalarType **, Layout>;
};

template <typename ScalarType, typename Layout>
struct ViewTypeRank<3, ScalarType, Layout> {
  using type = Kokkos::View<ScalarType ***, Layout>;
};

template <typename ScalarType, typename Layout>
struct ViewTypeRank<4, ScalarType, Layout> {
  using type = Kokkos::View<ScalarType ****, Layout>;
};

template <typename ScalarType, typename Layout>
struct ViewTypeRank<5, ScalarType, Layout> {
  using type = Kokkos::View<ScalarType *****, Layout>;
};

template <typename ScalarType, typename Layout>
struct ViewTypeRank<6, ScalarType, Layout> {
  using type = Kokkos::View<ScalarType ******, Layout>;
};


namespace Benchmark {

template <typename ExecutionSpace, typename ViewType, typename policy_type, bool nested_mdrange>
struct TeamMDRange_AMR_Benchmark {
  using layout          = ViewType::array_layout;
  using scalar_type     = ViewType::value_type;
  using execution_space = ExecutionSpace;
  using memory_space    = typename ExecutionSpace::memory_space;

  static constexpr Kokkos::Iterate iteration_pattern =
      LayoutToIterationPattern<layout>::pattern;
  static constexpr int rank = ViewType::rank();

  using policy_init_type = typename Kokkos::MDRangePolicy<
      execution_space,
      Kokkos::Rank<rank, iteration_pattern, iteration_pattern>>;
  using bound_type = typename policy_init_type::point_type;
  using tile_type  = typename policy_init_type::tile_type;

  ViewType m_view_A;
  ViewType m_view_B;

  bound_type m_lower_bounds;
  bound_type m_upper_bounds;
  tile_type m_tile;
  int m_vector_length = 1;
  Kokkos::Array<int, rank - 1> m_nested_upper_bounds;

  KOKKOS_INLINE_FUNCTION
  void operator()(int i0, int i1, int i2, int i3) const
    requires(!nested_mdrange && rank == 4)
  {
    i0 += 1;
    i1 += 1;
    i2 += 1;
    m_view_A(i0, i1, i2, i3) = -6.0 * m_view_B(i0, i1, i2, i3) + 
              (m_view_B(i0 + 1, i1, i2, i3) + m_view_B(i0 - 1, i1, i2, i3) +
               m_view_B(i0, i1 + 1, i2, i3) + m_view_B(i0, i1 - 1, i2, i3) +
               m_view_B(i0, i1, i2 + 1, i3) + m_view_B(i0, i1, i2 - 1, i3));
  }

  template <typename member_type>
    requires(nested_mdrange && rank == 4)
  KOKKOS_INLINE_FUNCTION void operator()(const member_type &team) const {
    const int team_idx = team.league_rank();
    // Iterate Left
    Kokkos::parallel_for(
        policy_type(team, m_nested_upper_bounds[0], m_nested_upper_bounds[1], m_nested_upper_bounds[2]),
        [=, *this](int i0, int i1, int i2) {
          auto i3 = team_idx;
          i2 += 1;
          i1 += 1;
          i0 += 1;
          m_view_A(i0, i1, i2, i3) = -6.0 * m_view_B(i0, i1, i2, i3) +
                     (m_view_B(i0 + 1, i1, i2, i3) + m_view_B(i0 - 1, i1, i2, i3) +
                      m_view_B(i0, i1 + 1, i2, i3) + m_view_B(i0, i1 - 1, i2, i3) +
                      m_view_B(i0, i1, i2 + 1, i3) + m_view_B(i0, i1, i2 - 1, i3));
        });
  }

  // Constructor to initialize views and bounds
  TeamMDRange_AMR_Benchmark(int n_block, int block_dim, tile_type tile) {
    for (int i = 0; i < rank - 1; ++i) {
      m_lower_bounds[i] = 0;
      m_upper_bounds[i] = block_dim;
      m_nested_upper_bounds[i] = m_upper_bounds[i];
    }
    m_lower_bounds[rank - 1] = 0;
    m_upper_bounds[rank - 1] = n_block;

    if constexpr(nested_mdrange) {
      m_vector_length = tile[0];
    } else {
      for (int i = 0; i < rank; ++i) {
        m_tile[i] = tile[i];
      }
    }

    block_dim += 2;

    m_view_A = ViewType("view_A", block_dim, block_dim, block_dim, n_block);
    m_view_B = ViewType("view_B", block_dim, block_dim, block_dim, n_block);

    Kokkos::parallel_for(policy_init_type(m_lower_bounds, {block_dim, block_dim, block_dim, n_block}),
                         Init(m_view_A, static_cast<scalar_type>(1.0)));
    Kokkos::parallel_for(policy_init_type(m_lower_bounds, {block_dim, block_dim, block_dim, n_block}),
                         Init(m_view_B, static_cast<scalar_type>(2.0)));
    execution_space().fence();
  }

  // Functor for initialization
  struct Init {
    ViewType mtx;
    scalar_type m_value;

    Init(const ViewType &tensor, const scalar_type &value)
        : mtx(tensor), m_value(value) {}

    template <typename... index>
    KOKKOS_INLINE_FUNCTION void operator()(index... idx) const {
      mtx(idx...) = m_value;
    }
  };

  struct Check {
    ViewType mtx;
    scalar_type m_value;

    Check(const ViewType &tensor, const scalar_type &value)
        : mtx(tensor), m_value(value) {}

    KOKKOS_INLINE_FUNCTION void operator()(int i0, int i1, int i2, int i3,
                                           int &num_err) const
      requires(rank == 4)
    {
      i0 += 1;
      i1 += 1;
      i2 += 1;
      if (mtx(i0, i1, i2, i3) != m_value) { ++num_err; }
    }

  };

  bool run_check() {
    if constexpr (nested_mdrange) {
      using policy_test_type = Kokkos::TeamPolicy<ExecutionSpace>;
      policy_test_type compute_policy(m_upper_bounds[rank - 1], Kokkos::AUTO(), m_vector_length);
      Kokkos::parallel_for("check_team", compute_policy, *this);
      execution_space().fence();
    } else {
      using policy_test_type = Kokkos::MDRangePolicy<
          Kokkos::Rank<rank, Kokkos::Iterate::Left, Kokkos::Iterate::Left>>;
      policy_test_type compute_policy(m_lower_bounds, m_upper_bounds, m_tile);
      Kokkos::parallel_for("check_mdrange", compute_policy, *this);
      execution_space().fence();
    }

    scalar_type excpected{0.0};

    Check checker(m_view_A, excpected);
    int num_err = 0;
    Kokkos::parallel_reduce(policy_init_type(m_lower_bounds, m_upper_bounds),
                            checker, num_err);
    return num_err == 0;
  }

  void run_benchmark(benchmark::State &state) {
    if constexpr (nested_mdrange) {
      using policy_test_type = Kokkos::TeamPolicy<ExecutionSpace>;
      policy_test_type compute_policy(m_upper_bounds[rank - 1], Kokkos::AUTO(), m_vector_length);
      for (auto _ : state) {
        Kokkos::Timer timer;
        Kokkos::parallel_for("bench_teamMDPolicy", compute_policy, *this);
        execution_space().fence();
        KokkosBenchmark::report_results(
            state, m_view_A, 6 , timer.seconds());
      }
    } else {
      using policy_test_type = Kokkos::MDRangePolicy<
          Kokkos::Rank<rank, Kokkos::Iterate::Left, Kokkos::Iterate::Left>>;
      policy_test_type compute_policy(m_lower_bounds, m_upper_bounds, m_tile);
      for (auto _ : state) {
        Kokkos::Timer timer;
        Kokkos::parallel_for("bench_MDPolicy", compute_policy, *this);
        execution_space().fence();
        KokkosBenchmark::report_results(
            state, m_view_A, 6, timer.seconds());
      }
    }
  }
};

template <typename BenchmarkType>
void run_amr_bench(benchmark::State &state) {
  int n_blocks = static_cast<int>(state.range(0));
  int block_dim = static_cast<int>(state.range(1));

  typename BenchmarkType::tile_type tile{};
  for (int i = 0; i < BenchmarkType::rank; ++i) {
    tile[i] = state.range(2 + i);
  }

  BenchmarkType amr_bench(n_blocks, block_dim, tile);

  if (!amr_bench.run_check()) {
    state.SkipWithError("Check failed");
    return;
  }

  amr_bench.run_benchmark(state);
}

template <int Rank, typename data_type>
void bench_MDRangePolicy(benchmark::State &state) {
  using ViewT = ViewTypeRank<Rank, data_type, Kokkos::LayoutLeft>::type;

  using bench_type =
      TeamMDRange_AMR_Benchmark<Kokkos::DefaultExecutionSpace, ViewT, void, false>;
  run_amr_bench<bench_type>(state);
}

template <int Rank, typename data_type>
void bench_TeamThreadRange(benchmark::State &state) {
  using execution_space = Kokkos::DefaultExecutionSpace;
  using ViewT       = ViewTypeRank<Rank, data_type, Kokkos::LayoutLeft>::type;
  using member_type = Kokkos::TeamPolicy<execution_space>::member_type;

  using policy_type = decltype(Kokkos::TeamThreadRange(
      std::declval<member_type const &>(), std::declval<int>()));

  using bench_type =
      TeamMDRange_AMR_Benchmark<execution_space, ViewT, policy_type, true>;
  run_amr_bench<bench_type>(state);
}

template <int Rank, typename data_type>
void bench_TeamVectorRange(benchmark::State &state) {
  using execution_space = Kokkos::DefaultExecutionSpace;
  using ViewT       = ViewTypeRank<Rank, data_type, Kokkos::LayoutLeft>::type;
  using member_type = Kokkos::TeamPolicy<execution_space>::member_type;

  using policy_type = decltype(Kokkos::TeamVectorRange(
      std::declval<member_type const &>(), std::declval<int>()));

  using bench_type =
      TeamMDRange_AMR_Benchmark<execution_space, ViewT, policy_type, true>;
  run_amr_bench<bench_type>(state);
}

template <int Rank, typename data_type>
void bench_teamThreadMDRange(benchmark::State &state) {
  using execution_space = Kokkos::DefaultExecutionSpace;
  using ViewT       = ViewTypeRank<Rank, data_type, Kokkos::LayoutLeft>::type;
  using member_type = Kokkos::TeamPolicy<execution_space>::member_type;
  using policy_type = Kokkos::TeamThreadMDRange<
      Kokkos::Rank<Rank - 1, Kokkos::Iterate::Left, Kokkos::Iterate::Left>,
      member_type>;

  using bench_type =
      TeamMDRange_AMR_Benchmark<execution_space, ViewT, policy_type, true>;
  run_amr_bench<bench_type>(state);
}

template <int Rank, typename data_type>
void bench_teamVectorMDRange(benchmark::State &state) {
  using execution_space = Kokkos::DefaultExecutionSpace;
  using ViewT       = ViewTypeRank<Rank, data_type, Kokkos::LayoutLeft>::type;
  using member_type = Kokkos::TeamPolicy<execution_space>::member_type;
  using policy_type = Kokkos::TeamVectorMDRange<
      Kokkos::Rank<Rank - 1, Kokkos::Iterate::Left, Kokkos::Iterate::Left>,
      member_type>;

  using bench_type =
      TeamMDRange_AMR_Benchmark<execution_space, ViewT, policy_type, true>;
  run_amr_bench<bench_type>(state);
}

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
#define N_BLOCKS 16384
#else
#define N_BLOCKS 4096
#endif

// List of blocks for stencil like computation
#define AMR_TEAM_BENCH_ARGS(function, n_block, block_dim, vector_length)  \
  BENCHMARK_TEMPLATE(function, 4, double)                                 \
      ->Args({n_block, block_dim, vector_length, 0, 0, 0})                \
      ->ArgNames({"n_blocks", "block_dim", "vector_length"})              \
      ->UseManualTime()                                                   \
      ->Unit(benchmark::kMillisecond);

#define AMR_MDRANGE_BENCH_ARGS(function, n_block, block_dim, t0, t1, t2, t3)     \
  BENCHMARK_TEMPLATE(function, 4, double)                                        \
      ->Args({n_block, block_dim, t0, t1, t2, t3})                               \
      ->ArgNames({"n_blocks", "block_dim", "tile0", "tile1", "tile2", "tile3"})  \
      ->UseManualTime()                                                          \
      ->Unit(benchmark::kMillisecond);

#define AMR_TEAM_POLICY_BENCH_ARGS(n_block, block_dim, vector_length)              \
  AMR_TEAM_BENCH_ARGS(bench_teamThreadMDRange, n_block, block_dim, vector_length)  \
  AMR_TEAM_BENCH_ARGS(bench_teamVectorMDRange, n_block, block_dim, vector_length)

#define AMR_MDRANGE_POLICY_BENCH_ARGS(n_block, block_dim, t0, t1, t2, t3)          \
  AMR_MDRANGE_BENCH_ARGS(bench_MDRangePolicy, n_block, block_dim, t0, t1, t2, t3) 

#define AMR_BLOCK_DIM_BENCH_ARGS(n_block, block_dim)          \
  AMR_MDRANGE_POLICY_BENCH_ARGS(n_block, block_dim, 0,0,0,0)  \
  AMR_MDRANGE_POLICY_BENCH_ARGS(n_block, block_dim, 8,8,4,1)  \
  AMR_TEAM_POLICY_BENCH_ARGS(n_block, block_dim, 1)           \
  AMR_TEAM_POLICY_BENCH_ARGS(n_block, block_dim, 2)           \
  AMR_TEAM_POLICY_BENCH_ARGS(n_block, block_dim, 4)           \
  AMR_TEAM_POLICY_BENCH_ARGS(n_block, block_dim, 8)           \
  AMR_TEAM_POLICY_BENCH_ARGS(n_block, block_dim, 16)          \
  AMR_TEAM_POLICY_BENCH_ARGS(n_block, block_dim, 32)

AMR_BLOCK_DIM_BENCH_ARGS(N_BLOCKS, 8)
AMR_BLOCK_DIM_BENCH_ARGS(N_BLOCKS, 12)
AMR_BLOCK_DIM_BENCH_ARGS(N_BLOCKS, 16)

}  // namespace Benchmark
