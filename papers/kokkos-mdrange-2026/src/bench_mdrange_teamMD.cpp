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

struct Tag_Add {};
struct Tag_Copy {};
struct Tag_2d_stencil_5pts {};
struct Tag_3d_stencil_7pts {};

template <typename Tag>
int tag_to_data_ratio() {
  if constexpr (std::is_same_v<Tag, Tag_Copy>) return 2;
  if constexpr (std::is_same_v<Tag, Tag_Add>) return 3;
  if constexpr (std::is_same_v<Tag, Tag_2d_stencil_5pts>) return 5;
  if constexpr (std::is_same_v<Tag, Tag_3d_stencil_7pts>) return 7;
  return 0;
}

namespace Benchmark {

template <typename ExecutionSpace, typename ViewType, typename policy_type,
          bool nested_mdrange, int VectorLength = 1>
struct TeamMDRange_Benchmark {
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

  ViewType m_view_A;
  ViewType m_view_B;
  ViewType m_view_C;

  bound_type m_lower_bounds;
  bound_type m_upper_bounds;
  Kokkos::Array<int, rank - 1> m_nested_upper_bounds;

  template <typename... Args>
    requires(!nested_mdrange)
  KOKKOS_INLINE_FUNCTION void operator()(Tag_Add, const Args... idx) const {
    m_view_C(idx...) = m_view_A(idx...) + m_view_B(idx...);
  }

  template <typename... Args>
    requires(!nested_mdrange)
  KOKKOS_INLINE_FUNCTION void operator()(Tag_Copy, const Args... idx) const {
    m_view_C(idx...) = m_view_A(idx...);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(Tag_2d_stencil_5pts, int i0, int i1) const
    requires(!nested_mdrange && rank == 2)
  {
    i0++;
    i1++;
    m_view_A(i0, i1) =
        0.25 * (m_view_B(i0 + 1, i1) + m_view_B(i0 - 1, i1) +
                m_view_B(i0, i1 + 1) + m_view_B(i0, i1 - 1) + m_view_B(i0, i1));
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(Tag_3d_stencil_7pts, int i0, int i1, int i2) const
    requires(!nested_mdrange && rank == 3)
  {
    i0++;
    i1++;
    i2++;
    m_view_A(i0, i1, i2) =
        0.25 * (m_view_B(i0 + 1, i1, i2) + m_view_B(i0 - 1, i1, i2) +
                m_view_B(i0, i1 + 1, i2) + m_view_B(i0, i1 - 1, i2) +
                m_view_B(i0, i1, i2 + 1) + m_view_B(i0, i1, i2 - 1) +
                m_view_B(i0, i1, i2));
  }

  template <typename member_type, std::size_t... Idxs>
  KOKKOS_INLINE_FUNCTION void apply_nested(Tag_Add, const member_type &team,
                                           std::index_sequence<Idxs...>) const {
    const int team_idx = team.league_rank();
    Kokkos::parallel_for(policy_type(team, m_nested_upper_bounds[Idxs]...),
                         [=, *this](const auto... idx) {
                           m_view_C(idx..., team_idx) =
                               m_view_A(idx..., team_idx) +
                               m_view_B(idx..., team_idx);
                         });
  }

  template <typename member_type, std::size_t... Idxs>
  KOKKOS_INLINE_FUNCTION void apply_nested(Tag_Copy, const member_type &team,
                                           std::index_sequence<Idxs...>) const {
    const int team_idx = team.league_rank();
    Kokkos::parallel_for(policy_type(team, m_nested_upper_bounds[Idxs]...),
                         [=, *this](const auto... idx) {
                           m_view_C(idx..., team_idx) =
                               m_view_A(idx..., team_idx);
                         });
  }

  template <typename member_type>
    requires(nested_mdrange)
  KOKKOS_INLINE_FUNCTION void operator()(Tag_Add,
                                         const member_type &team) const {
    apply_nested(Tag_Add{}, team, std::make_index_sequence<rank - 1>{});
  }

  template <typename member_type>
    requires(nested_mdrange)
  KOKKOS_INLINE_FUNCTION void operator()(Tag_Copy,
                                         const member_type &team) const {
    apply_nested(Tag_Copy{}, team, std::make_index_sequence<rank - 1>{});
  }

  template <typename member_type>
    requires(nested_mdrange && rank == 2)
  KOKKOS_INLINE_FUNCTION void operator()(Tag_2d_stencil_5pts,
                                         const member_type &team) const {
    const int team_idx = team.league_rank();
    // Iterate Left
    Kokkos::parallel_for(
        policy_type(team, m_nested_upper_bounds[0]), [=, *this](int i0) {
          auto i1 = team_idx + 1;
          i0 += 1;
          m_view_A(i0, i1) =
              0.25 *
              (m_view_B(i0 + 1, i1) + m_view_B(i0 - 1, i1) +
               m_view_B(i0, i1 + 1) + m_view_B(i0, i1 - 1) + m_view_B(i0, i1));
        });
  }

  template <typename member_type>
    requires(nested_mdrange && rank == 3)
  KOKKOS_INLINE_FUNCTION void operator()(Tag_3d_stencil_7pts,
                                         const member_type &team) const {
    const int team_idx = team.league_rank();
    // Iterate Left
    Kokkos::parallel_for(
        policy_type(team, m_nested_upper_bounds[0], m_nested_upper_bounds[1]),
        [=, *this](int i0, int i1) {
          auto i2 = team_idx + 1;
          i1 += 1;
          i0 += 1;
          m_view_A(i0, i1, i2) =
              0.25 * (m_view_B(i0 + 1, i1, i2) + m_view_B(i0 - 1, i1, i2) +
                      m_view_B(i0, i1 + 1, i2) + m_view_B(i0, i1 - 1, i2) +
                      m_view_B(i0, i1, i2 + 1) + m_view_B(i0, i1, i2 - 1) +
                      m_view_B(i0, i1, i2));
        });
  }

  // Create test views
  template <typename array_type, std::size_t... Idxs>
  ViewType create_test_view(const char *name, array_type dims,
                            std::index_sequence<Idxs...>) {
    std::string view_name(name);
    return ViewType(view_name, dims[Idxs]...);
  }

  // Constructor to initialize views and bounds
  template <typename array_type, typename Tag>
  TeamMDRange_Benchmark(array_type dims, Tag) {
    for (int i = 0; i < rank; ++i) {
      m_lower_bounds[i] = 0;
      m_upper_bounds[i] = dims[i];
      if constexpr (std::is_same_v<Tag, Tag_2d_stencil_5pts> ||
                    std::is_same_v<Tag, Tag_3d_stencil_7pts>) {
        dims[i] += 2;
      }
    }
    for (int i = 0; i < rank - 1; ++i) {
      m_nested_upper_bounds[i] = m_upper_bounds[i];
    }

    m_view_A =
        create_test_view("view_A", dims, std::make_index_sequence<rank>{});
    m_view_B =
        create_test_view("view_B", dims, std::make_index_sequence<rank>{});
    m_view_C =
        create_test_view("view_C", dims, std::make_index_sequence<rank>{});

    Kokkos::parallel_for(policy_init_type(m_lower_bounds, m_upper_bounds),
                         Init(m_view_A, static_cast<scalar_type>(1.0)));
    Kokkos::parallel_for(policy_init_type(m_lower_bounds, m_upper_bounds),
                         Init(m_view_B, static_cast<scalar_type>(2.0)));
    Kokkos::parallel_for(policy_init_type(m_lower_bounds, m_upper_bounds),
                         Init(m_view_C, static_cast<scalar_type>(0.0)));
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

    KOKKOS_INLINE_FUNCTION void operator()(int i0, int i1, int &num_err) const
      requires(rank == 2)
    {
      if (mtx(i0, i1) != m_value) ++num_err;
    }

    KOKKOS_INLINE_FUNCTION void operator()(int i0, int i1, int i2,
                                           int &num_err) const
      requires(rank == 3)
    {
      if (mtx(i0, i1, i2) != m_value) ++num_err;
    }

    KOKKOS_INLINE_FUNCTION void operator()(int i0, int i1, int i2, int i3,
                                           int &num_err) const
      requires(rank == 4)
    {
      if (mtx(i0, i1, i2, i3) != m_value) ++num_err;
    }

    KOKKOS_INLINE_FUNCTION void operator()(int i0, int i1, int i2, int i3,
                                           int i4, int &num_err) const
      requires(rank == 5)
    {
      if (mtx(i0, i1, i2, i3, i4) != m_value) ++num_err;
    }
  };

  template <typename Tag>
  bool run_check(Tag) {
    if constexpr (nested_mdrange) {
      using policy_test_type = Kokkos::TeamPolicy<ExecutionSpace, Tag>;
      policy_test_type compute_policy(m_upper_bounds[rank - 1], Kokkos::AUTO(),
                                      VectorLength);
      Kokkos::parallel_for("check_team", compute_policy, *this);
      execution_space().fence();
    } else {
      using policy_test_type = Kokkos::MDRangePolicy<
          Kokkos::Rank<rank, Kokkos::Iterate::Left, Kokkos::Iterate::Left>,
          Tag>;
      policy_test_type compute_policy(m_lower_bounds, m_upper_bounds);
      Kokkos::parallel_for("check_mdrange", compute_policy, *this);
      execution_space().fence();
    }

    scalar_type excpected{0};
    if constexpr (std::is_same_v<Tag, Tag_Copy>) {
      excpected = 1;
    } else if constexpr (std::is_same_v<Tag, Tag_Add>) {
      excpected = 3;
    }

    Check checker(m_view_C, excpected);
    int num_err = 0;
    Kokkos::parallel_reduce(policy_init_type(m_lower_bounds, m_upper_bounds),
                            checker, num_err);
    return num_err == 0;
  }

  template <typename Tag>
  void run_benchmark(benchmark::State &state, Tag) {
    if constexpr (nested_mdrange) {
      using policy_test_type = Kokkos::TeamPolicy<ExecutionSpace, Tag>;
      policy_test_type compute_policy(m_upper_bounds[rank - 1], Kokkos::AUTO(),
                                      VectorLength);
      for (auto _ : state) {
        Kokkos::Timer timer;
        Kokkos::parallel_for("bench_teamMDPolicy", compute_policy, *this);
        execution_space().fence();
        KokkosBenchmark::report_results(
            state, m_view_C, tag_to_data_ratio<Tag>(), timer.seconds());
      }
    } else {
      using policy_test_type = Kokkos::MDRangePolicy<
          Kokkos::Rank<rank, Kokkos::Iterate::Left, Kokkos::Iterate::Left>,
          Tag>;
      policy_test_type compute_policy(m_lower_bounds, m_upper_bounds);
      for (auto _ : state) {
        Kokkos::Timer timer;
        Kokkos::parallel_for("bench_MDPolicy", compute_policy, *this);
        execution_space().fence();
        KokkosBenchmark::report_results(
            state, m_view_C, tag_to_data_ratio<Tag>(), timer.seconds());
      }
    }
  }
};

template <typename BenchmarkType, typename Tag>
void run_stream_bench(benchmark::State &state) {
  int M = static_cast<int>(state.range(0));
  Kokkos::Array<int, BenchmarkType::rank> dims;
  for (std::size_t i = 0; i < BenchmarkType::rank; i++) {
    dims[i] = M;
  }

  BenchmarkType stream_bench(dims, Tag{});
  if (!stream_bench.run_check(Tag{})) {
    state.SkipWithError("Check failed");
    return;
  }
  stream_bench.run_benchmark(state, Tag{});
}

template <typename Tag, int Rank, typename data_type>
void run_MDRange(benchmark::State &state) {
  using ViewT = ViewTypeRank<Rank, data_type, Kokkos::LayoutLeft>::type;

  using bench_type =
      TeamMDRange_Benchmark<Kokkos::DefaultExecutionSpace, ViewT, void, false>;
  run_stream_bench<bench_type, Tag>(state);
}

template <typename Tag, int Rank, typename data_type>
void run_teamThreadRange(benchmark::State &state) {
  using execution_space = Kokkos::DefaultExecutionSpace;
  using ViewT       = ViewTypeRank<Rank, data_type, Kokkos::LayoutLeft>::type;
  using member_type = Kokkos::TeamPolicy<execution_space, Tag>::member_type;

  using policy_type = decltype(Kokkos::TeamThreadRange(
      std::declval<member_type const &>(), std::declval<int>()));

  using bench_type =
      TeamMDRange_Benchmark<execution_space, ViewT, policy_type, true>;
  run_stream_bench<bench_type, Tag>(state);
}

template <typename Tag, int Rank, typename data_type>
void run_teamVectorRange(benchmark::State &state) {
  using execution_space = Kokkos::DefaultExecutionSpace;
  using ViewT       = ViewTypeRank<Rank, data_type, Kokkos::LayoutLeft>::type;
  using member_type = Kokkos::TeamPolicy<execution_space, Tag>::member_type;

  using policy_type = decltype(Kokkos::TeamVectorRange(
      std::declval<member_type const &>(), std::declval<int>()));

  using bench_type =
      TeamMDRange_Benchmark<execution_space, ViewT, policy_type, true>;
  run_stream_bench<bench_type, Tag>(state);
}

template <typename Tag, int Rank, typename data_type>
void run_teamThreadMDRange(benchmark::State &state) {
  using execution_space = Kokkos::DefaultExecutionSpace;
  using ViewT       = ViewTypeRank<Rank, data_type, Kokkos::LayoutLeft>::type;
  using member_type = Kokkos::TeamPolicy<execution_space, Tag>::member_type;
  using policy_type = Kokkos::TeamThreadMDRange<
      Kokkos::Rank<Rank - 1, Kokkos::Iterate::Left, Kokkos::Iterate::Left>,
      member_type>;

  using bench_type =
      TeamMDRange_Benchmark<execution_space, ViewT, policy_type, true>;
  run_stream_bench<bench_type, Tag>(state);
}

template <typename Tag, int Rank, typename data_type>
void run_teamVectorMDRange(benchmark::State &state) {
  using execution_space = Kokkos::DefaultExecutionSpace;
  using ViewT       = ViewTypeRank<Rank, data_type, Kokkos::LayoutLeft>::type;
  using member_type = Kokkos::TeamPolicy<execution_space, Tag>::member_type;
  using policy_type = Kokkos::TeamVectorMDRange<
      Kokkos::Rank<Rank - 1, Kokkos::Iterate::Left, Kokkos::Iterate::Left>,
      member_type>;

  using bench_type =
      TeamMDRange_Benchmark<execution_space, ViewT, policy_type, true, 32>;
  run_stream_bench<bench_type, Tag>(state);
}

#define STREAM_BENCH_ARGS(function, tag, rank, size) \
  BENCHMARK_TEMPLATE(function, tag, rank, double)    \
      ->Arg(size)                                    \
      ->ArgNames({"M"})                              \
      ->UseManualTime()                              \
      ->Unit(benchmark::kMillisecond);

STREAM_BENCH_ARGS(run_MDRange, Tag_2d_stencil_5pts, 2, 10240)
STREAM_BENCH_ARGS(run_teamThreadRange, Tag_2d_stencil_5pts, 2, 10240)
STREAM_BENCH_ARGS(run_teamVectorRange, Tag_2d_stencil_5pts, 2, 10240)

#define STREAM_BENCH_POLICY(tag, rank, size)                \
  STREAM_BENCH_ARGS(run_MDRange, tag, rank, size)           \
  STREAM_BENCH_ARGS(run_teamThreadMDRange, tag, rank, size) \
  STREAM_BENCH_ARGS(run_teamVectorMDRange, tag, rank, size)

STREAM_BENCH_POLICY(Tag_3d_stencil_7pts, 3, 512)

#define STREAM_BENCH_RANK(tag)     \
  STREAM_BENCH_POLICY(tag, 3, 512) \
  STREAM_BENCH_POLICY(tag, 4, 96)  \
  STREAM_BENCH_POLICY(tag, 5, 32)

STREAM_BENCH_RANK(Tag_Copy)
STREAM_BENCH_RANK(Tag_Add)

}  // namespace Benchmark
