#include <classes.hpp>

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);

  {
    Foo_1* f_1 = (Foo_1*) Kokkos::kokkos_malloc(sizeof(Foo_1));
    Foo_2* f_2 = (Foo_2*) Kokkos::kokkos_malloc(sizeof(Foo_2));
    Foo** f =  (Foo**) Kokkos::kokkos_malloc(sizeof(Foo*)*2);


    // Placement new construction of objects on the device
    Kokkos::parallel_for(
        "Set", 1, KOKKOS_LAMBDA(const int&) {
          new (f_1) Foo_1();
          f[0]   = (Foo*) f_1;
          f_1->set_values(2,4);
          new (f_2) Foo_2();
          f[1] = (Foo*) f_2;
          f_2->set_values(3,5);
        });

    // CHeck that the values are correct on the device
    int errors;
    Kokkos::parallel_reduce(
        "CheckValues", 1,
        KOKKOS_LAMBDA(const int& , int& lsum) {
          if(f[0]->value() != 2*1000000+4) lsum++;
          if(f[1]->value() != 3*1000000+5) lsum++;
    },errors);
    printf("Errors Initial: %i\n", errors);

    // Extract Nested Data Classes
    Kokkos::View<Data*> data_v("data",2);
    auto h_data_v = Kokkos::create_mirror_view(data_v);

    Kokkos::parallel_for("CopyToData",1,
        KOKKOS_LAMBDA(const int&) {
          data_v(0) = f_1->data;
          data_v(1) = f_2->data;
    });

    Kokkos::deep_copy(h_data_v, data_v);

    // Modify Data on host
    h_data_v(0).val_derived=8;
    h_data_v(1).val_derived=9;

    // Copy it up again
    Kokkos::deep_copy(data_v, h_data_v);

    Kokkos::parallel_for("CopyToData",1,
        KOKKOS_LAMBDA(const int&) {
          f_1->data = data_v(0);
          f_2->data = data_v(1);
    });

    // Access data as used by base class
    Kokkos::parallel_reduce(
        "CheckValues", 1,
        KOKKOS_LAMBDA(const int& , int& lsum) {
          if(f[0]->value() != 2*1000000+8) lsum++;
          if(f[1]->value() != 3*1000000+9) lsum++;
    },errors);
    printf("Errors After Copy: %i\n", errors);

    Kokkos::parallel_for(
        "DestroyObjects", 1, KOKKOS_LAMBDA(const int&) {
          f[0]->~Foo();
          f[1]->~Foo();
        });
    Kokkos::kokkos_free(f_1);
    Kokkos::kokkos_free(f_2);
    Kokkos::kokkos_free(f);
  }

  Kokkos::finalize();
}
