# Apps in consideration for Europar-2024

Instructions to build the apps with the `europar2024` branch of my Kokkos fork. The fork is up-to date with the upstream Kokkos `develop` branch from today.
Need some `Makefile.kokkos` changes to build the apps if Makefile build system is used.

## CMAKE

### Build Kokkos
Building OpenMPTarget backend of Kokkos with `clang` compiler
```bash
git clone [https://github.com/rgayatri23/kokkos.git](https://github.com/rgayatri23/kokkos.git)
cd kokkos
git checkout europar2024
mkdir build && cd build

cmake \
  -DCMAKE_CXX_COMPILER=clang++                     \
  -DCMAKE_BUILD_TYPE=Release                       \
  -DCMAKE_INSTALL_PREFIX=$path-to-kokkos-install   \
  -DKokkos_ENABLE_OPENMPTARGET=ON                  \
  -DKokkos_ARCH_AMPERE80=ON                        \
  ../
```

### Building the APPS

* AXPBY
```bash
cd axpby-dot/
cmake \
  -DCMAKE_CXX_COMPILER=clang++          \
  -DCMAKE_BUILD_TYPE=Release            \
  -DKokkos_ROOT=$path-to-kokkos-install \
  ../
```

* CGSolve
cd cgsolve/
cmake \
  -DCMAKE_CXX_COMPILER=clang++          \
  -DCMAKE_BUILD_TYPE=Release            \
  -DKokkos_ROOT=$path-to-kokkos-install \
  ../
```

* TestSNAP
```bash
cd TestSNAP/
cmake \
  -DCMAKE_CXX_COMPILER=clang++          \
  -DCMAKE_BUILD_TYPE=Release            \
  -DKokkos_ROOT=$path-to-kokkos-install \
  -Dref_data=14                         \
  ../
```

## Makefile

Edit the first line of `Makefile.inc` to point to the Kokkos source. 
The options set by default are for NVIDIA A100 architecture running with the OpenMPTarget backend of Kokkos.

```
