#include<atomic_benchmark.hpp>

void measure_atomics_int(int N, int M, int R, int K) {
  measure_atomics<int>(N, M, R, K);
}
