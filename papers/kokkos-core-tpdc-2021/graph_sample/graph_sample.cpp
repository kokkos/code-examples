
#include <Kokkos_Core.hpp>
#include <Kokkos_Graph.hpp>

#define N 1024
#define I 100

using Vector_t = Kokkos::View<double*>;
using Scalar_t = Kokkos::View<double>;

template <class ExecSpace>
struct Axpby {
  Vector_t x, y;
  double alpha, beta;
  KOKKOS_FUNCTION
  void operator()(const int& i) const noexcept {
    x(i) = alpha * x(i) + beta * y(i);
  }
};

template <class ExecSpace, class T>
struct Dot {
  Vector_t x, y;
  KOKKOS_FUNCTION
  void operator()(const int& i, T& lsum) const noexcept { lsum += x(i) * y(i); }
};

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  Vector_t x("x", N), y("y", N), z("z", N);
  Scalar_t dotp("dotp");
  double alpha, beta, gamma;
  using EXECSPACE     = Kokkos::DefaultExecutionSpace;
  using axpby_functor = Axpby<EXECSPACE>;
  using dotp_functor  = Dot<EXECSPACE, double>;
  EXECSPACE ex{};

  // Set up
  Kokkos::deep_copy(ex, dotp, 0);
  Kokkos::deep_copy(ex, x, 1);
  Kokkos::deep_copy(ex, y, 2);
  Kokkos::deep_copy(ex, z, 3);
  ex.fence();

  // Construct graph
  auto graph = Kokkos::Experimental::create_graph(ex, [&](auto root) {
    auto f_xpy = root.then_parallel_for(N, axpby_functor{x, y, alpha, beta});
    auto f_zpy = root.then_parallel_for(N, axpby_functor{z, y, gamma, beta});
    auto ready = when_all(f_xpy, f_zpy);
    ready.then_parallel_reduce(N, dotp_functor{x, z}, dotp);
  });

  // Submit many times
  int iters = I;
  while (--iters > 0) {
    graph.submit();
    ex.fence(); /*foo(dotp)*/
  }
  return 0;
}
