#include<atomic_benchmark.hpp>

void measure_atomics_double(int N, int M, int R, int K) {
  measure_atomics<double>(N, M, R, K);
}
