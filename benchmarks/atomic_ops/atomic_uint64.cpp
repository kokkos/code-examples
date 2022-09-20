#include<atomic_benchmark.hpp>

void measure_atomics_uint64(int N, int M, int R, int K) {
  measure_atomics<uint64_t>(N, M, R, K);
}
