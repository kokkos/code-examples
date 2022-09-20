#include<atomic_benchmark.hpp>

void measure_atomics_unsigned(int N, int M, int R, int K) {
  measure_atomics<unsigned>(N, M, R, K);
}
