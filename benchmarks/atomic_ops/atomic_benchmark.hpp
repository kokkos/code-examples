#include <Kokkos_Core.hpp>
#include <cmath>
#include <Kokkos_Random.hpp>

#define BinaryAtomic( name ) \
struct name { \
  template<class T, class U> \
  KOKKOS_FUNCTION \
  static void op(T* dest, U upd) { \
    (void) Kokkos::atomic_##name(dest,upd); \
  } \
};

struct Min {
  template<class T, class U>
  KOKKOS_FUNCTION
  static void op(T* dest, U upd) {
    (void) Kokkos::atomic_min(dest,upd);
  }
};
struct Max {
  template<class T, class U>
  KOKKOS_FUNCTION
  static void op(T* dest, U upd) {
    (void) Kokkos::atomic_max(dest,upd);
  }
};

BinaryAtomic(add)
BinaryAtomic(fetch_add)
BinaryAtomic(add_fetch)
BinaryAtomic(sub)
BinaryAtomic(fetch_sub)
BinaryAtomic(sub_fetch)
BinaryAtomic(fetch_min)
BinaryAtomic(min_fetch)
BinaryAtomic(fetch_max)
BinaryAtomic(max_fetch)

template<class T, class Op>
void measure_single_atomic(int R, bool mode_offset, Kokkos::View<T*> dest, Kokkos::View<T*> src, Kokkos::View<int*>(idx), Op) {
  Kokkos::fence();
  Kokkos::Timer timer;
  for(int r=0; r<R; r++)
  Kokkos::parallel_for(src.extent(0), KOKKOS_LAMBDA(int i) {
    int ii = mode_offset ? (i + idx(i))%dest.extent(0) : idx(i)%dest.extent(0);
    Op::op(&dest(ii), src(i));
  });
  Kokkos::fence();
  double time = timer.seconds();
  double gops = src.extent(0)/time/1.e9*R;
  printf("Scalar: %s Op: %s GOPS: %e Config: %i %i %i\n",typeid(T).name(), typeid(Op).name(), gops, (int) src.extent(0), (int) dest.extent(0), mode_offset?1:0);
}

template<class T>
void measure_atomics(int N, int M, int R, int K) {
  bool mo = K>=0;
  Kokkos::View<T*> dest("dest",M);
  Kokkos::View<T*> src("src",N);
  Kokkos::View<int*> idx("idx",N);

  Kokkos::Random_XorShift64_Pool<> g(1931);
  Kokkos::fill_random(idx,g,mo?K:M);

  measure_single_atomic(R, mo, dest, src, idx, add());
  measure_single_atomic(R, mo, dest, src, idx, fetch_add());
  measure_single_atomic(R, mo, dest, src, idx, add_fetch());
  measure_single_atomic(R, mo, dest, src, idx, sub());
  measure_single_atomic(R, mo, dest, src, idx, fetch_sub());
  measure_single_atomic(R, mo, dest, src, idx, sub_fetch());
  measure_single_atomic(R, mo, dest, src, idx, Min());
  measure_single_atomic(R, mo, dest, src, idx, fetch_min());
  measure_single_atomic(R, mo, dest, src, idx, min_fetch());
  measure_single_atomic(R, mo, dest, src, idx, Max());
  measure_single_atomic(R, mo, dest, src, idx, fetch_max());
  measure_single_atomic(R, mo, dest, src, idx, max_fetch());
}

void measure_atomics_int(int,int,int,int);
void measure_atomics_unsigned(int,int,int,int);
void measure_atomics_int64(int,int,int,int);
void measure_atomics_uint64(int,int,int,int);
void measure_atomics_float(int,int,int,int);
void measure_atomics_double(int,int,int,int);
