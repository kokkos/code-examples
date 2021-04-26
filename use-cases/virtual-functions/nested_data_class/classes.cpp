#include <classes.hpp>

KOKKOS_FUNCTION
Foo::Foo() { val = 0; }

KOKKOS_FUNCTION
Foo_1::Foo_1() { val = 1; data.val_derived = 1; }

KOKKOS_FUNCTION
int Foo_1::value() { return val * 1000000 + data.val_derived; }

KOKKOS_FUNCTION
Foo_2::Foo_2() { val = 2; data.val_derived = 2; }

KOKKOS_FUNCTION
int Foo_2::value() { return val * 1000000 + data.val_derived; }
