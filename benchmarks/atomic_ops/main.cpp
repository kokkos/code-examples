#include <Kokkos_Core.hpp>
#include <atomic_benchmark.hpp>

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    int N = argc > 1 ? atoi(argv[1]) : 10000000;
    int M = argc > 2 ? atoi(argv[2]) : 100000;
    int R = argc > 3 ? atoi(argv[3]) : 10;
    int K = argc > 4 ? atoi(argv[4]) : -1;

    measure_atomics_int(N,M,R,K);
    measure_atomics_unsigned(N,M,R,K);
    measure_atomics_int64(N,M,R,K);
    measure_atomics_uint64(N,M,R,K);
    measure_atomics_float(N,M,R,K);
    measure_atomics_double(N,M,R,K);

  }
  Kokkos::finalize();
}
