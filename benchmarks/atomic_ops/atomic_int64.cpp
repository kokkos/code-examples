#include<atomic_benchmark.hpp>

void measure_atomics_int64(int N, int M, int R, int K) {
  measure_atomics<int64_t>(N, M, R, K);
}
