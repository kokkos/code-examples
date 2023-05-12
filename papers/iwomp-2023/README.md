# Code examples for paper on OpenMPTarget backend of Kokkos - IWOMP 2023

## Build 

Edit the `KOKKOS_PATH` in Makefile.inc to point to a valid Kokkos source repo.
In the individual app directories follow the below `make` options.

### Native backends

### CUDA on NVIDIA A100

```shell
make -j8 backend=cuda arch=A100
```

### HIP on AMD MI250x

```shell
make -j8 backend=hip arch=MI250x
```


## OpenMPTarget backend

### clang on NVIDIA A100

```shell
make -j8 backend=cuda arch=A100 comp=clang
```

### nvhpc on NVIDIA A100

```shell
make -j8 backend=cuda arch=A100 comp=nvhpc
```

### amdclang on AMD MI250x

```shell
make -j8 backend=cuda arch=MI250x comp=amdclang
```
