#include <Kokkos_Core.hpp>
#include <cmath>

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    int N = argc > 1 ? atoi(argv[1]) : 1000000;
    int R = argc > 2 ? atoi(argv[2]) : 10;

    auto exec = Kokkos::DefaultExecutionSpace();
    Kokkos::Profiling::pushRegion("Init");
    Kokkos::View<double*> a("A",N);
    exec.fence("First Fence\n");

    Kokkos::deep_copy(a,1);
    Kokkos::Profiling::popRegion();
    Kokkos::Profiling::pushRegion("Global Fence");
    Kokkos::fence();
    Kokkos::Profiling::popRegion();

  }
  Kokkos::finalize();
}
