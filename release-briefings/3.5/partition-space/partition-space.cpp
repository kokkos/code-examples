#include <Kokkos_Core.hpp>
#include <cmath>

struct Functor {
  Kokkos::View<double*> x;
  int R;
  KOKKOS_FUNCTION
  void operator() (const int i) const {
    double val = x[i];
    for(int r=0; r<R; r++) {
      val += val*(1./(r+i));
    }
    x[i] = val;
  }
};


void test(int N0, int N1, int R0, int R1, bool one_exec, bool print = true) {

Kokkos::View<double*> d_A("A", N0);
auto h_A = Kokkos::create_mirror_view(d_A);
Kokkos::deep_copy(d_A,1.0);

Kokkos::View<double*> d_B("B", N1);
auto h_B = Kokkos::create_mirror_view(d_B);
Kokkos::deep_copy(d_B,2.0);

auto FunctorA = Functor{d_A,R0};
auto FunctorB = Functor{d_B,R1};


// Get a handle to the original instance
auto original = Kokkos::DefaultExecutionSpace();
Kokkos::Timer timer;
// Split it 2 ways
auto instances = Kokkos::Experimental::partition_space(original, 1, 1);
double init_secs = timer.seconds();

timer.reset();
auto exec_0 = instances[0];
auto exec_1 = instances[one_exec?0:1];
// Dispatch FunctorA1 and FunctorB so that they execute
// potentially at the same time
Kokkos::parallel_for("A1", Kokkos::RangePolicy<>(exec_0, 0, N0), FunctorA);
Kokkos::parallel_for("B",  Kokkos::RangePolicy<>(exec_1, 0, N1), FunctorB);

// Enqueue a copy from the result of FunctorA to the host
Kokkos::deep_copy(exec_0, h_A, d_A);

// Wait for just FunctorA1 and the deep_copy to finish
instances[0].fence();

// Dispatch more work to instance[0]
Kokkos::parallel_for("A2", Kokkos::RangePolicy<>(exec_0, 0, N0), FunctorA);

// Wait for A2 and B
Kokkos::fence();

double time = timer.seconds();

if(print)
  printf("Time init: %3.1lfus Time run: %3.1lfus\n",init_secs*1000000, time*1000000);
}


int main(int argc, char* argv[]) {
  // Long running saturates GPU each  100000 100000 50000 100000
  // Long running doesn't saturate GPU 10000 10000 50000 100000

  Kokkos::initialize(argc, argv);
  {
    int N0 = argc > 1 ? atoi(argv[1]) : 10000;
    int N1 = argc > 2 ? atoi(argv[2]) : 30000;
    int R0 = argc > 3 ? atoi(argv[3]) : 500;
    int R1 = argc > 4 ? atoi(argv[4]) : 1500;
    test(N0,N1,R0,R1,true,false);

    printf("Run with N0: %i N1: %i R0: %i R1: %i\n", N0, N1, R0, R1);
    printf("\nRunning with a single instance:\n");
    test(N0,N1,R0,R1,true);
    printf("\nRunning with two instances:\n");
    test(N0,N1,R0,R1,false);

  }
  Kokkos::finalize();
}
