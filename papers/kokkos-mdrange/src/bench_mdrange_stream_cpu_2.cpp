// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <functional>
#include <iostream>
#include <numeric>
#include <limits>

#include <benchmark/benchmark.h>

#include <Kokkos_Core.hpp>

int main(int argc, char **argv) {
  Kokkos::initialize(Kokkos::InitializationSettings().set_num_threads(4));
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

namespace Benchmark {

struct Tag_Set {};
struct Tag_Copy {};
struct Tag_Scale {};
struct Tag_Add {};
struct Tag_Triad {};

template <typename Tag>
int tag_to_data_ratio() {
  if constexpr (std::is_same_v<Tag, Tag_Set>) return 1;
  if constexpr (std::is_same_v<Tag, Tag_Copy>) return 2;
  if constexpr (std::is_same_v<Tag, Tag_Scale>) return 2;
  if constexpr (std::is_same_v<Tag, Tag_Add>) return 3;
  if constexpr (std::is_same_v<Tag, Tag_Triad>) return 3;
  return 0;
}

template <int Rank, typename ScalarType, typename Layout, typename MemorySpace>
struct ViewTypeRank {
  using type = void;
};

template <typename ScalarType, typename Layout, typename MemorySpace>
struct ViewTypeRank<1, ScalarType, Layout, MemorySpace> {
  using type = Kokkos::View<ScalarType *, Layout, MemorySpace>;
};

template <typename ScalarType, typename Layout, typename MemorySpace>
struct ViewTypeRank<2, ScalarType, Layout, MemorySpace> {
  using type = Kokkos::View<ScalarType **, Layout, MemorySpace>;
};

template <typename ScalarType, typename Layout, typename MemorySpace>
struct ViewTypeRank<3, ScalarType, Layout, MemorySpace> {
  using type = Kokkos::View<ScalarType ***, Layout, MemorySpace>;
};

template <typename ScalarType, typename Layout, typename MemorySpace>
struct ViewTypeRank<4, ScalarType, Layout, MemorySpace> {
  using type = Kokkos::View<ScalarType ****, Layout, MemorySpace>;
};

template <typename ScalarType, typename Layout, typename MemorySpace>
struct ViewTypeRank<5, ScalarType, Layout, MemorySpace> {
  using type = Kokkos::View<ScalarType *****, Layout, MemorySpace>;
};

template <typename ScalarType, typename Layout, typename MemorySpace>
struct ViewTypeRank<6, ScalarType, Layout, MemorySpace> {
  using type = Kokkos::View<ScalarType ******, Layout, MemorySpace>;
};

template <int Rank, typename ExecutionSpace,
          typename IndexType = Kokkos::IndexType<int>, typename Tag = void>
struct policy_selector {
  using preferred_layout = typename ExecutionSpace::array_layout;
  static const Kokkos::Iterate outer_iter =
      Kokkos::Impl::layout_iterate_type_selector<
          preferred_layout>::outer_iteration_pattern;
  static const Kokkos::Iterate inner_iter =
      Kokkos::Impl::layout_iterate_type_selector<
          preferred_layout>::inner_iteration_pattern;
  using type = Kokkos::MDRangePolicy<ExecutionSpace,
                                     Kokkos::Rank<Rank, outer_iter, inner_iter>,
                                     IndexType, Tag>;
};

template <typename PolicyType>
struct bound_type_selector {
  using type = typename PolicyType::point_type;
};

// Functor for stream test (set, copy, scale, add, triad).
// The problem size is N^6, meaning that each view will have the size of N^6
// whatever the rank is.
template <typename ExecutionSpace, int Rank, typename ScalarType = double,
          typename IndexType = Kokkos::IndexType<int>>
struct MDStreamTest {
  using scalar_type      = ScalarType;
  using execution_space  = ExecutionSpace;
  using memory_space     = typename ExecutionSpace::memory_space;
  using preferred_layout = typename ExecutionSpace::array_layout;
  using view_type = typename ViewTypeRank<Rank, scalar_type, preferred_layout,
                                          memory_space>::type;
  using policy_init_type =
      typename policy_selector<Rank, ExecutionSpace, IndexType, void>::type;
  using bound_type = typename bound_type_selector<policy_init_type>::type;

  view_type m_view_A;
  view_type m_view_B;
  view_type m_view_C;
  ScalarType m_scalar;

  int m_N;
  bound_type m_lower_bounds;
  bound_type m_upper_bounds;
  bound_type m_tile_dims;

  MDStreamTest(const int N) {
    static_assert(Rank >= 1 && Rank <= 6,
                  "MDStreamTest: Only ranks 1 to 6 supported");

    m_view_A = create_test_view("MDStreamTest::view_A", N);
    m_view_B = create_test_view("MDStreamTest::view_B", N);
    m_view_C = create_test_view("MDStreamTest::view_C", N);
    m_scalar = static_cast<scalar_type>(2.718281828);

    for (int i = 0; i < Rank; ++i) {
      m_lower_bounds[i] = 0;
      m_upper_bounds[i] = m_view_A.extent(i);
      m_tile_dims[i]    = 8;
    }

    if constexpr (policy_selector<Rank, ExecutionSpace, IndexType,
                                  void>::inner_iter == Kokkos::Iterate::Right) {
      m_tile_dims[Rank - 1] = 256;
    } else {
      m_tile_dims[0] = 256;
    }

    policy_init_type init_policy(m_lower_bounds, m_upper_bounds);
    Kokkos::parallel_for(init_policy,
                         Init(m_view_A, static_cast<ScalarType>(1.0)));
    Kokkos::parallel_for(init_policy,
                         Init(m_view_B, static_cast<ScalarType>(2.0)));
    Kokkos::parallel_for(init_policy,
                         Init(m_view_C, static_cast<ScalarType>(3.0)));
    execution_space().fence();
  }

  // Tagged operator()
  KOKKOS_INLINE_FUNCTION void operator()(Tag_Set, const int i) const {
    m_view_A(i) = static_cast<ScalarType>(m_scalar);
  }

  KOKKOS_INLINE_FUNCTION void operator()(Tag_Set, const int i,
                                         const int j) const {
    m_view_A(i, j) = static_cast<ScalarType>(m_scalar);
  }

  KOKKOS_INLINE_FUNCTION void operator()(Tag_Set, const int i, const int j,
                                         const int k) const {
    m_view_A(i, j, k) = static_cast<ScalarType>(m_scalar);
  }

  KOKKOS_INLINE_FUNCTION void operator()(Tag_Set, const int i, const int j,
                                         const int k, const int l) const {
    m_view_A(i, j, k, l) = static_cast<ScalarType>(m_scalar);
  }

  KOKKOS_INLINE_FUNCTION void operator()(Tag_Set, const int i, const int j,
                                         const int k, const int l,
                                         const int m) const {
    m_view_A(i, j, k, l, m) = static_cast<ScalarType>(m_scalar);
  }

  KOKKOS_INLINE_FUNCTION void operator()(Tag_Set, const int i, const int j,
                                         const int k, const int l, const int m,
                                         const int n) const {
    m_view_A(i, j, k, l, m, n) = static_cast<ScalarType>(m_scalar);
  }

  KOKKOS_INLINE_FUNCTION void operator()(Tag_Copy, const int i) const {
    m_view_B(i) = m_view_A(i);
  }

  KOKKOS_INLINE_FUNCTION void operator()(Tag_Copy, const int i,
                                         const int j) const {
    m_view_B(i, j) = m_view_A(i, j);
  }

  KOKKOS_INLINE_FUNCTION void operator()(Tag_Copy, const int i, const int j,
                                         const int k) const {
    m_view_B(i, j, k) = m_view_A(i, j, k);
  }

  KOKKOS_INLINE_FUNCTION void operator()(Tag_Copy, const int i, const int j,
                                         const int k, const int l) const {
    m_view_B(i, j, k, l) = m_view_A(i, j, k, l);
  }

  KOKKOS_INLINE_FUNCTION void operator()(Tag_Copy, const int i, const int j,
                                         const int k, const int l,
                                         const int m) const {
    m_view_B(i, j, k, l, m) = m_view_A(i, j, k, l, m);
  }

  KOKKOS_INLINE_FUNCTION void operator()(Tag_Copy, const int i, const int j,
                                         const int k, const int l, const int m,
                                         const int n) const {
    m_view_B(i, j, k, l, m, n) = m_view_A(i, j, k, l, m, n);
  }

  KOKKOS_INLINE_FUNCTION void operator()(Tag_Scale, const int i) const {
    m_view_B(i) = m_scalar * m_view_A(i);
  }

  KOKKOS_INLINE_FUNCTION void operator()(Tag_Scale, const int i,
                                         const int j) const {
    m_view_B(i, j) = m_scalar * m_view_A(i, j);
  }

  KOKKOS_INLINE_FUNCTION void operator()(Tag_Scale, const int i, const int j,
                                         const int k) const {
    m_view_B(i, j, k) = m_scalar * m_view_A(i, j, k);
  }

  KOKKOS_INLINE_FUNCTION void operator()(Tag_Scale, const int i, const int j,
                                         const int k, const int l) const {
    m_view_B(i, j, k, l) = m_scalar * m_view_A(i, j, k, l);
  }

  KOKKOS_INLINE_FUNCTION void operator()(Tag_Scale, const int i, const int j,
                                         const int k, const int l,
                                         const int m) const {
    m_view_B(i, j, k, l, m) = m_scalar * m_view_A(i, j, k, l, m);
  }

  KOKKOS_INLINE_FUNCTION void operator()(Tag_Scale, const int i, const int j,
                                         const int k, const int l, const int m,
                                         const int n) const {
    m_view_B(i, j, k, l, m, n) = m_scalar * m_view_A(i, j, k, l, m, n);
  }

  KOKKOS_INLINE_FUNCTION void operator()(Tag_Add, const int i) const {
    m_view_C(i) = m_view_A(i) + m_view_B(i);
  }

  KOKKOS_INLINE_FUNCTION void operator()(Tag_Add, const int i,
                                         const int j) const {
    m_view_C(i, j) = m_view_A(i, j) + m_view_B(i, j);
  }

  KOKKOS_INLINE_FUNCTION void operator()(Tag_Add, const int i, const int j,
                                         const int k) const {
    m_view_C(i, j, k) = m_view_A(i, j, k) + m_view_B(i, j, k);
  }

  KOKKOS_INLINE_FUNCTION void operator()(Tag_Add, const int i, const int j,
                                         const int k, const int l) const {
    m_view_C(i, j, k, l) = m_view_A(i, j, k, l) + m_view_B(i, j, k, l);
  }

  KOKKOS_INLINE_FUNCTION void operator()(Tag_Add, const int i, const int j,
                                         const int k, const int l,
                                         const int m) const {
    m_view_C(i, j, k, l, m) = m_view_A(i, j, k, l, m) + m_view_B(i, j, k, l, m);
  }

  KOKKOS_INLINE_FUNCTION void operator()(Tag_Add, const int i, const int j,
                                         const int k, const int l, const int m,
                                         const int n) const {
    m_view_C(i, j, k, l, m, n) =
        m_view_A(i, j, k, l, m, n) + m_view_B(i, j, k, l, m, n);
  }

  KOKKOS_INLINE_FUNCTION void operator()(Tag_Triad, const int i) const {
    m_view_C(i) = m_view_A(i) + m_scalar * m_view_B(i);
  }

  KOKKOS_INLINE_FUNCTION void operator()(Tag_Triad, const int i,
                                         const int j) const {
    m_view_C(i, j) = m_view_A(i, j) + m_scalar * m_view_B(i, j);
  }

  KOKKOS_INLINE_FUNCTION void operator()(Tag_Triad, const int i, const int j,
                                         const int k) const {
    m_view_C(i, j, k) = m_view_A(i, j, k) + m_scalar * m_view_B(i, j, k);
  }

  KOKKOS_INLINE_FUNCTION void operator()(Tag_Triad, const int i, const int j,
                                         const int k, const int l) const {
    m_view_C(i, j, k, l) =
        m_view_A(i, j, k, l) + m_scalar * m_view_B(i, j, k, l);
  }

  KOKKOS_INLINE_FUNCTION void operator()(Tag_Triad, const int i, const int j,
                                         const int k, const int l,
                                         const int m) const {
    m_view_C(i, j, k, l, m) =
        m_view_A(i, j, k, l, m) + m_scalar * m_view_B(i, j, k, l, m);
  }

  KOKKOS_INLINE_FUNCTION void operator()(Tag_Triad, const int i, const int j,
                                         const int k, const int l, const int m,
                                         const int n) const {
    m_view_C(i, j, k, l, m, n) =
        m_view_A(i, j, k, l, m, n) + m_scalar * m_view_B(i, j, k, l, m, n);
  }

  // Create test views of size N^6
  view_type create_test_view(const char *name, int dim) {
    long N1 = dim;
    long N2 = N1 * N1;
    long N3 = N2 * N1;
    long N6 = N3 * N3;
    std::string view_name(name);
    if constexpr (Rank == 1) {
      return view_type(view_name, N6);
    } else if constexpr (Rank == 2) {
      return view_type(view_name, N3, N3);
    } else if constexpr (Rank == 3) {
      return view_type(view_name, N2, N2, N2);
    } else if constexpr (Rank == 4) {
      return view_type(view_name, N2, N2, N1, N1);
    } else if constexpr (Rank == 5) {
      return view_type(view_name, N2, N1, N1, N1, N1);
    } else if constexpr (Rank == 6) {
      return view_type(view_name, N1, N1, N1, N1, N1, N1);
    }
  }

  // Functor for initialization
  struct Init {
    view_type m_tensor;
    scalar_type m_value;

    Init(const view_type &tensor, const scalar_type &value)
        : m_tensor(tensor), m_value(value) {}

    template <typename... Indices>
    KOKKOS_INLINE_FUNCTION void operator()(Indices... indices) const {
      m_tensor(indices...) = m_value;
    }
  };

  template <typename Tag>
  void run_test(benchmark::State &state) {
    using policy_test_type =
        typename policy_selector<Rank, ExecutionSpace, IndexType, Tag>::type;

    policy_test_type compute_policy(m_lower_bounds, m_upper_bounds,
                                    m_tile_dims);

    const int data_ratio = tag_to_data_ratio<Tag>();

    for (auto _ : state) {
      Kokkos::Timer timer;
      Kokkos::parallel_for(compute_policy, *this);
      execution_space().fence();
      KokkosBenchmark::report_results(state, m_view_A, data_ratio,
                                      timer.seconds());
    }
  }

  void test_set(benchmark::State &state) { run_test<Tag_Set>(state); }

  void test_copy(benchmark::State &state) { run_test<Tag_Copy>(state); }

  void test_scale(benchmark::State &state) { run_test<Tag_Scale>(state); }

  void test_add(benchmark::State &state) { run_test<Tag_Add>(state); }

  void test_triad(benchmark::State &state) { run_test<Tag_Triad>(state); }
};

template <int Rank>
void MDRangePolicy_Copy(benchmark::State &state) {
  int N = static_cast<int>(state.range(0));

  MDStreamTest<Kokkos::DefaultExecutionSpace, Rank, float> stream_bench(N);
  stream_bench.test_copy(state);
}

template <int Rank>
void MDRangePolicy_Set(benchmark::State &state) {
  int N = static_cast<int>(state.range(0));

  MDStreamTest<Kokkos::DefaultExecutionSpace, Rank, float> stream_bench(N);
  stream_bench.test_set(state);
}

template <int Rank>
void MDRangePolicy_Scale(benchmark::State &state) {
  int N = static_cast<int>(state.range(0));

  MDStreamTest<Kokkos::DefaultExecutionSpace, Rank, float> stream_bench(N);
  stream_bench.test_scale(state);
}

template <int Rank>
void MDRangePolicy_Add(benchmark::State &state) {
  int N = static_cast<int>(state.range(0));

  MDStreamTest<Kokkos::DefaultExecutionSpace, Rank, float> stream_bench(N);
  stream_bench.test_add(state);
}

template <int Rank>
void MDRangePolicy_Triad(benchmark::State &state) {
  int N = static_cast<int>(state.range(0));

  MDStreamTest<Kokkos::DefaultExecutionSpace, Rank, float> stream_bench(N);
  stream_bench.test_triad(state);
}

// Small size for CPU backends
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || \
    defined(KOKKOS_ENABLE_SYCL)
#define MDRANGE_BENCHMARK_ARG_SIZE 24
#else
#define MDRANGE_BENCHMARK_ARG_SIZE 16
#endif

// Macros to generate benchmarks
#define MDRANGE_BENCHMARK_ARGS(BENCH_FUNCTION, RANKS) \
  BENCHMARK_TEMPLATE(BENCH_FUNCTION, RANKS)           \
      ->Arg(MDRANGE_BENCHMARK_ARG_SIZE)               \
      ->UseManualTime()                               \
      ->Unit(benchmark::kMillisecond);

#define MDRANGE_MAKE_BENCHMARK(BENCH_FUNCTION) \
  MDRANGE_BENCHMARK_ARGS(BENCH_FUNCTION, 1)    \
  MDRANGE_BENCHMARK_ARGS(BENCH_FUNCTION, 2)    \
  MDRANGE_BENCHMARK_ARGS(BENCH_FUNCTION, 3)    \
  MDRANGE_BENCHMARK_ARGS(BENCH_FUNCTION, 4)    \
  MDRANGE_BENCHMARK_ARGS(BENCH_FUNCTION, 5)    \
  MDRANGE_BENCHMARK_ARGS(BENCH_FUNCTION, 6)

/*
#define MDRANGE_MAKE_BENCHMARK(BENCH_FUNCTION) \
  MDRANGE_BENCHMARK_ARGS(BENCH_FUNCTION, 2)    \
  MDRANGE_BENCHMARK_ARGS(BENCH_FUNCTION, 3)
*/
MDRANGE_MAKE_BENCHMARK(MDRangePolicy_Copy)
MDRANGE_MAKE_BENCHMARK(MDRangePolicy_Set)
MDRANGE_MAKE_BENCHMARK(MDRangePolicy_Add)
MDRANGE_MAKE_BENCHMARK(MDRangePolicy_Scale)
MDRANGE_MAKE_BENCHMARK(MDRangePolicy_Triad)

#undef MDRANGE_BENCHMARK_ARG_SIZE
#undef MDRANGE_BENCHMARK_ARGS
#undef MDRANGE_MAKE_BENCHMARK

}  // namespace Benchmark
