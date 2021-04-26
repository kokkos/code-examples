#ifndef KOKKOS_EXAMPLE_VIRTUAL_FUNCTIONS_CLASSES_HPP
#define KOKKOS_EXAMPLE_VIRTUAL_FUNCTIONS_CLASSES_HPP

#include <Kokkos_Core.hpp>

class Foo {
 protected:
  int val;

 public:
  KOKKOS_FUNCTION
  Foo();

  KOKKOS_FUNCTION
  virtual int value() { return 0; };

  KOKKOS_FUNCTION
  virtual ~Foo() {}
};

class Foo_1 : public Foo {
 protected:
  int val_derived;
 public:
  KOKKOS_FUNCTION
  Foo_1();

  KOKKOS_FUNCTION
  int value();

  KOKKOS_FUNCTION
  void set_values(int val_, int val_derived_) {
    val = val_;
    val_derived = val_derived_;
  }
};

class Foo_2 : public Foo {
 protected:
  int val_derived;
 public:
  KOKKOS_FUNCTION
  Foo_2();

  KOKKOS_FUNCTION
  int value();

  KOKKOS_FUNCTION
  void set_values(int val_, int val_derived_) {
    val = val_;
    val_derived = val_derived_;
  }
};

#endif  // KOKKOS_EXAMPLE_VIRTUAL_FUNCTIONS_CLASSES_HPP
