### Explore IndexType impact for mdspan accessors

This example explores the impact of using size_t as the offset type in `mdspan` accessor `access` functions.

To that end we are using a custom accessor which is a copy of `default_accessor` with a template parameter for the offset type.
The code is designed to stress that access function as much as possible.

### Compiling

In some directory:
```
git clone https://github.com/kokkos/kokkos
git clone https://github.com/kokkos/mdspan
```

Modify the `Makefile` of this example to point `KOKKOS_PATH` and `MDSPAN_PATH` to the repository directories.

This example is using a basic Kokkos Makefile, you have to choose architecture and backend. 

```
# compile for NVIDIA A100
make -j KOKKOS_DEVICES=Cuda KOKKOS_ARCH=Ampere80
# compile for AMD MI2xx
make -j KOKKOS_DEVICES=HIP KOKKOS_ARCH=Vega90A CXX=hipcc
```
