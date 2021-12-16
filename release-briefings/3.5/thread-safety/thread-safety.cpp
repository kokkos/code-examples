#include <Kokkos_Core.hpp>
#include <cmath>
#include <thread>

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

void foo(int N, int R, Kokkos::View<double*> a) {
  Functor f{a,R};
  Kokkos::parallel_for("Hello", N, f);
}
void bar(int N, int R, Kokkos::View<double*> a, Kokkos::DefaultExecutionSpace exec) {
  Functor f{a,R};
  Kokkos::parallel_for("Hello", Kokkos::RangePolicy<>(exec,0,N), f);
}

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    int N = argc > 1 ? atoi(argv[1]) : 1000000;
    int R = argc > 2 ? atoi(argv[2]) : 10;

    Kokkos::View<double*> a1("A1",N);
    Kokkos::View<double*> a2("A2",N);
    Kokkos::View<double*> a3("A3",N);
    Kokkos::deep_copy(a1,1.0);
    Kokkos::deep_copy(a2,2.0);
    Kokkos::deep_copy(a3,3.0);

    Kokkos::Timer timer;
    {
      foo(N,R,a1);
      foo(N,R,a2);
      foo(N,R,a3);
      Kokkos::fence();
    }
    double time_serial = timer.seconds();
    timer.reset();
    {
      std::thread t1(foo,N,R,a1);
      std::thread t2(foo,N,R,a2);
      std::thread t3(foo,N,R,a3);

      t1.join();
      t2.join();
      t3.join();
      Kokkos::fence();
    }
    double time_same = timer.seconds();

    auto instances =
      Kokkos::Experimental::partition_space(Kokkos::DefaultExecutionSpace(), 1, 1, 1);
    Kokkos::fence();
    timer.reset();
    {
      std::thread t1(bar,N,R,a1,instances[0]);
      std::thread t2(bar,N,R,a2,instances[1]);
      std::thread t3(bar,N,R,a3,instances[2]);

      t1.join();
      t2.join();
      t3.join();
      Kokkos::fence();
    }
    double time_different = timer.seconds();

    printf("TimeSerial: %lf TimeSame: %lf TimeDifferent: %lf\n",time_serial, time_same, time_different);
  }
  Kokkos::finalize();
}
