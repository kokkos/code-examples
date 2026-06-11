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

struct Tag_Naive_default {};
struct Tag_Naive_8x8 {};
struct Tag_Naive_16x16 {};
// struct Tag_Naive_32x16 {};
// struct Tag_Naive_16x32 {};
struct Tag_Flattening {};

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

template <typename Tag>
Kokkos::Array<std::int64_t, 2> get_benchmark_tile(const Tag) {
  if constexpr (std::is_same_v<Tag, Tag_Naive_default>) {
    return {0, 0};
  } else if constexpr (std::is_same_v<Tag, Tag_Naive_8x8>) {
    return {8, 8};
  } else if constexpr (std::is_same_v<Tag, Tag_Naive_16x16>) {
    return {16, 16};
  } else {
    return {0, 0};
  }
}

namespace Benchmark {

//  A [M * K]
//  B [K * N]
//  C [M * N]
template <typename ExecutionSpace, typename Layout, typename ScalarType = double,
          typename IndexType = Kokkos::IndexType<uint32_t>>
struct MDRange_gemm_naive {
  static constexpr Kokkos::Iterate iteration_pattern = LayoutToIterationPattern<Layout>::pattern;
  using scalar_type      = ScalarType;
  using execution_space  = ExecutionSpace;
  using memory_space     = typename ExecutionSpace::memory_space;
  using view_type        = typename Kokkos::View<scalar_type**, Layout, memory_space>;
  using policy_init_type = typename Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<2, iteration_pattern,iteration_pattern>, IndexType>;
  using bound_type       = typename policy_init_type::point_type;

  view_type m_view_A;
  view_type m_view_B;
  view_type m_view_C;

  int m_M;
  int m_N;
  int m_K;
  bound_type m_lower_bounds;
  bound_type m_upper_bounds;

  KOKKOS_INLINE_FUNCTION
  void operator()(Tag_Naive_default, const int i, const int j) const {
    scalar_type sum = 0.0;
    for (int k = 0; k < m_K; ++k) {
      sum += m_view_A(i, k) * m_view_B(k, j);
    }
    m_view_C(i, j) = sum;
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(Tag_Naive_8x8, const int i, const int j) const {
    scalar_type sum = 0.0;
    for (int k = 0; k < m_K; ++k) {
      sum += m_view_A(i, k) * m_view_B(k, j);
    }
    m_view_C(i, j) = sum;
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(Tag_Naive_16x16, const int i, const int j) const {
    scalar_type sum = 0.0;
    for (int k = 0; k < m_K; ++k) {
      sum += m_view_A(i, k) * m_view_B(k, j);
    }
    m_view_C(i, j) = sum;
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(Tag_Flattening, const int r) const {
    if constexpr (std::is_same_v<Layout, Kokkos::LayoutLeft>) {
      int i = r % m_upper_bounds[0];
      int j = r / m_upper_bounds[0];
      double sum = 0.0;
      for (int k = 0; k < m_K; ++k) {
        sum += m_view_A(i, k) * m_view_B(k, j);
      }
      m_view_C(i, j) = sum;
    } else {
      double sum = 0.0;
      int j = r % m_upper_bounds[1];
      int i = r / m_upper_bounds[1];
      for (int k = 0; k < m_K; ++k) {
        sum += m_view_A(i, k) * m_view_B(k, j);
      }
      m_view_C(i, j) = sum;
    }
  }

  // Create test views
  view_type create_test_view(const char *name, int dim_M, int dim_N) {
    std::string view_name(name);
    return view_type(view_name, dim_M, dim_N);
  }

  // Constructor to initialize views and bounds
  MDRange_gemm_naive(const int M, const int N, const int K) {
    m_M = M;
    m_N = N;
    m_K = K;

    m_lower_bounds = bound_type{0, 0};
    m_upper_bounds = bound_type{M, N};

    m_view_A = create_test_view("MDRange_gemm_naive::view_A", M, K);
    m_view_B = create_test_view("MDRange_gemm_naive::view_B", K, N);
    m_view_C = create_test_view("MDRange_gemm_naive::view_C", M, N);

    Kokkos::parallel_for(policy_init_type(m_lower_bounds,{M, K}),
                         Init(m_view_A, static_cast<ScalarType>(1.0)));
    Kokkos::parallel_for(policy_init_type(m_lower_bounds,{K, N}),
                         Init(m_view_B, static_cast<ScalarType>(2.0)));
    Kokkos::parallel_for(policy_init_type(m_lower_bounds,{M, N}),
                         Init(m_view_C, static_cast<ScalarType>(0.0)));
    execution_space().fence();
  }

  // Functor for initialization
  struct Init {
    view_type mtx;
    scalar_type m_value;

    Init(const view_type &tensor, const scalar_type &value)
        : mtx(tensor), m_value(value) {}
    template <typename index_type>
    KOKKOS_INLINE_FUNCTION void operator()(index_type i, index_type j) const {
      mtx(i, j) = m_value;
    }
  };

  template <typename Tag>
  bool run_check(const Tag) {
    if constexpr (std::is_same_v<Tag, Tag_Flattening>) {
      using policy_test_type = Kokkos::RangePolicy<ExecutionSpace, IndexType, Tag_Flattening>;
      policy_test_type compute_policy(0, m_M * m_N);
      Kokkos::parallel_for(compute_policy, *this);
      execution_space().fence();
    } else {
      using policy_test_type = Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<2, iteration_pattern, iteration_pattern>, IndexType, Tag>;
      bound_type tile = get_benchmark_tile(Tag{});
      policy_test_type compute_policy(m_lower_bounds, m_upper_bounds, tile);
      Kokkos::parallel_for(compute_policy, *this);
      execution_space().fence();
    }

    // common check
    auto host_view_C = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), m_view_C);
    if constexpr (!std::is_same_v<scalar_type, Kokkos::Experimental::half_t>) {
      for (int i = 0; i < m_M; ++i) {
        for (int j = 0; j < m_N; ++j) {
          if (host_view_C(i, j) != static_cast<scalar_type>(m_K * 2.0)) {
            std::cerr << "Error at (" << i << ", " << j << ")\n";
            return false;
          }
        }
      }
    }
    return true;
  }

  template <typename Tag>
  void run_test(benchmark::State &state, const Tag) {
    const int data_ratio = (m_M * m_N + m_M * m_K + m_K * m_N) / (m_M * m_N);
    const unsigned long flops = 2UL * m_M * m_N * m_K;

    if constexpr (std::is_same_v<Tag, Tag_Flattening>) {
      using policy_test_type = Kokkos::RangePolicy<ExecutionSpace, IndexType, Tag>;
      policy_test_type compute_policy(0, m_M * m_N);
      for (auto _ : state) {
        Kokkos::Timer timer;
        Kokkos::parallel_for(compute_policy, *this);
        execution_space().fence();
        state.counters[KokkosBenchmark::benchmark_fom("GFLOP/s")] =
            benchmark::Counter(flops / 1'000'000'000, benchmark::Counter::kIsIterationInvariantRate);
        KokkosBenchmark::report_results(state, m_view_C, data_ratio, timer.seconds());
      }
    } else {
      using policy_test_type = Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<2, iteration_pattern, iteration_pattern>, IndexType, Tag>;
      bound_type tile = get_benchmark_tile(Tag{});
      policy_test_type compute_policy(m_lower_bounds, m_upper_bounds, tile);
      for (auto _ : state) {
        Kokkos::Timer timer;
        Kokkos::parallel_for(compute_policy, *this);
        execution_space().fence();
        state.counters[KokkosBenchmark::benchmark_fom("GFLOP/s")] =
            benchmark::Counter(flops / 1'000'000'000, benchmark::Counter::kIsIterationInvariantRate);
        KokkosBenchmark::report_results(state, m_view_C, data_ratio, timer.seconds());
      }
    }
  }
};

template <typename Tag, typename Layout, typename data_type>
void run_mdrange_gemm(benchmark::State &state) {
  int M = static_cast<int>(state.range(0));
  int N = static_cast<int>(state.range(1));
  int K = static_cast<int>(state.range(2));

  MDRange_gemm_naive<Kokkos::DefaultExecutionSpace, Layout, data_type> gemm_bench(M, N, K);
  if (!gemm_bench.run_check(Tag{})) {
    state.SkipWithError("Check failed");
  }
  gemm_bench.run_test(state, Tag{});
}


#define GEMM_BENCH_ARGS(function, layout, data_type, tag, size)  \
BENCHMARK_TEMPLATE(function, tag, layout, data_type)             \
  ->Args({size, size, size})                                     \
  ->ArgNames({"M", "N", "K"})                                    \
  ->UseManualTime()                                              \
  ->Unit(benchmark::kMillisecond);

#define GEMM_BENCH_SIZE(function, layout, data_type, tag)  \
GEMM_BENCH_ARGS(function, layout, data_type, tag, 2048)    \
GEMM_BENCH_ARGS(function, layout, data_type, tag, 4096)    \
GEMM_BENCH_ARGS(function, layout, data_type, tag, 6144)    \
GEMM_BENCH_ARGS(function, layout, data_type, tag, 8192)    \
GEMM_BENCH_ARGS(function, layout, data_type, tag, 10240)   \
GEMM_BENCH_ARGS(function, layout, data_type, tag, 12288)   \
// GEMM_BENCH_ARGS(function, layout, data_type, tag, 14336)   \
// GEMM_BENCH_ARGS(function, layout, data_type, tag, 16384)   \

#define GEMM_BENCH_LAYOUT_DATA_TYPE(function, tag)                                  \
GEMM_BENCH_SIZE(function, Kokkos::LayoutLeft, double, tag)                          \
GEMM_BENCH_SIZE(function, Kokkos::LayoutLeft, float,  tag)                          \
GEMM_BENCH_SIZE(function, Kokkos::LayoutLeft, Kokkos::Experimental::half_t,  tag)  \

GEMM_BENCH_LAYOUT_DATA_TYPE(run_mdrange_gemm, Tag_Naive_default)
GEMM_BENCH_LAYOUT_DATA_TYPE(run_mdrange_gemm, Tag_Naive_8x8)
GEMM_BENCH_LAYOUT_DATA_TYPE(run_mdrange_gemm, Tag_Naive_16x16)
GEMM_BENCH_LAYOUT_DATA_TYPE(run_mdrange_gemm, Tag_Flattening)

}  // namespace Benchmark
