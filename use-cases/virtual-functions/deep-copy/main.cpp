#include <classes.hpp>

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);

  {
    int N = 10;
    Foo_1* f_1 = (Foo_1*) Kokkos::kokkos_malloc(sizeof(Foo_1)*N);
    Foo_2* f_2 = (Foo_2*) Kokkos::kokkos_malloc(sizeof(Foo_2)*N);
    Foo_1* h_f_1 = (Foo_1*) Kokkos::kokkos_malloc<Kokkos::DefaultHostExecutionSpace>(sizeof(Foo_1)*N);
    Foo_2* h_f_2 = (Foo_2*) Kokkos::kokkos_malloc<Kokkos::DefaultHostExecutionSpace>(sizeof(Foo_2)*N);
    Foo** f =  (Foo**) Kokkos::kokkos_malloc(sizeof(Foo*)*N*2);

    // Placement new construction of objects on the device
    Kokkos::parallel_for(
        "Set", N, KOKKOS_LAMBDA(const int& i) {
          new (&f_1[i]) Foo_1();
          f[2*i]   = (Foo*) &f_1[i];
          f_1[i].set_values(2*i,4*i);
          new (&f_2[i]) Foo_2();
          f[2*i+1] = (Foo*) &f_2[i];
          f_2[i].set_values(2*i+1,4*i+1);
        });

    // CHeck that the values are correct on the device
    int errors;
    Kokkos::parallel_reduce(
        "CheckValues", N,
        KOKKOS_LAMBDA(const int& i, int& lsum) {
          if(f[2*i]->value()  !=2*i*1000000+4*i) lsum++;
          if(f[2*i+1]->value()!=(2*i+1)*1000000+4*i+1) lsum++;
    },errors);
    printf("Errors Initial: %i\n", errors);

    // We can create unmanaged views around those guys
    Kokkos::View<Foo_1*> f_1_v(f_1,N);
    Kokkos::View<Foo_2*> f_2_v(f_2,N);
    Kokkos::View<Foo_1*, Kokkos::DefaultHostExecutionSpace> h_f_1_v(h_f_1,N);
    Kokkos::View<Foo_2*, Kokkos::DefaultHostExecutionSpace> h_f_2_v(h_f_2,N);


    // We can deep copy those guys
    // Note: h_f_1_v virtual table is now unusable
    //       - the pointer points to device function
    //       - but the object itself is only accessible on the host
    Kokkos::deep_copy(h_f_1_v, f_1_v);
    Kokkos::deep_copy(h_f_2_v, f_2_v);

    // We can still set the values though on the properly typed objects!
    Kokkos::parallel_for(
        "SetHost", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,N), KOKKOS_LAMBDA(const int& i) {
          h_f_1_v(i).set_values(4*i,8*i);
          h_f_2_v(i).set_values(4*i+1,8*i+1);
        });

    // We can copy back
    Kokkos::deep_copy(f_1_v, h_f_1_v);
    Kokkos::deep_copy(f_2_v, h_f_2_v);
    Kokkos::parallel_reduce(
        "CheckValues", N,
        KOKKOS_LAMBDA(const int& i, int& lsum) {
          if(f[2*i]->value()  !=4*i*1000000+8*i) lsum++;
          if(f[2*i+1]->value()!=(4*i+1)*1000000+8*i+1) lsum++;
    },errors);
    printf("Errors After Copy: %i\n", errors);

    Kokkos::parallel_for(
        "DestroyObjects", 2*N, KOKKOS_LAMBDA(const int& i) {
          f[i]->~Foo();
        });
    Kokkos::kokkos_free(f_1);
    Kokkos::kokkos_free(f_2);
    Kokkos::kokkos_free(h_f_1);
    Kokkos::kokkos_free(h_f_2);
    Kokkos::kokkos_free(f);
  }

  Kokkos::finalize();
}
