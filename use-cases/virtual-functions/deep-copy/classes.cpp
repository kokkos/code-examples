#include <classes.hpp>

KOKKOS_FUNCTION
Foo::Foo() { val = 0; }

KOKKOS_FUNCTION
Foo_1::Foo_1() { val = 1; val_derived = 1; }

KOKKOS_FUNCTION
int Foo_1::value() { return val * 1000000 + val_derived; }

KOKKOS_FUNCTION
Foo_2::Foo_2() { val = 2; val_derived = 2; }

KOKKOS_FUNCTION
int Foo_2::value() { return val * 1000000 + val_derived; }
