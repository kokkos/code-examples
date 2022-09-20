#include<atomic_benchmark.hpp>

void measure_atomics_float(int N, int M, int R, int K) {
  measure_atomics<float>(N, M, R, K);
}
