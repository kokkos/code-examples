/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_ScatterView.hpp>

#if !defined(USE_OMP) && !defined(USE_CUDA) && !defined(USE_KOKKOS) && !(USE_OMPT)
#define USE_KOKKOS
#endif

using particle_pos_t = Kokkos::View<double*[3]>;

#if defined(USE_CUDA)
template<class LAMBDA>
__global__ void cuda_kernel(size_t N, const LAMBDA f) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i<N) f(i);
}
#endif


using particle_pos_t = Kokkos::View<double*[3]>;

template<class particle_val_t, class grid_t, class grid_sv_t>
void discretize(double dx, particle_pos_t pos, particle_val_t value, grid_t grid, grid_sv_t grid_sv) {
#if defined(USE_KOKKOS)
  grid_sv.reset();
  Kokkos::parallel_for("Deposit", pos.extent(0), KOKKOS_LAMBDA(int i) {
    auto grid_access = grid_sv.access();
    int x = pos(i,0)/dx;
    int y = pos(i,1)/dx;
    int z = pos(i,2)/dx;
    grid_access(x,y,z)+=value(i);
  });
  Kokkos::Experimental::contribute(grid,grid_sv);
#else
  auto pos_p = pos.data();
  auto N = pos.extent(0);
  auto val_p = value.data();
  auto grid_p = grid.data();
  auto grid_y = grid.extent(1);
#if defined(USE_OMP)
  auto grid_z = grid.extent(2);
  auto grid_x = grid.extent(0);
  typename particle_val_t::value_type* grid_ptrs[64];
  for(int t=0; t<Kokkos::OpenMP().concurrency(); t++)
    grid_ptrs[t] = Kokkos::subview(grid_sv,t,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL).data();

  // Reset buffer data to zero with correct first touch
  #pragma omp parallel
  {
    int t = omp_get_thread_num();
    for(int x = 0; x<grid_x; x++)
    for(int y = 0; y<grid_y; y++)
    for(int z = 0; z<grid_z; z++) {
      grid_ptrs[t][(x*grid_y + y)*grid_z + z] = 0.0;;
    }
  }

  // Run Kernel
  #pragma omp parallel for
  for(int i = 0; i<N; i++) {
    int x = pos_p[i*3]/dx;
    int y = pos_p[i*3+1]/dx;
    int z = pos_p[i*3+2]/dx;
    // Matching LayoutRight indexing of Kokkos CPU variant
    grid_ptrs[omp_get_thread_num()][(x*grid_y + y)*grid_z + z] += val_p[i];
  }

  // Sum up contributions to original view, likely with bad memory affinity
  #pragma omp parallel for collapse(3)
  for(int x = 0; x<grid_x; x++)
  for(int y = 0; y<grid_y; y++)
  for(int z = 0; z<grid_z; z++) {
    for(int t=0; t<omp_get_num_threads(); t++)
       grid_p[(x*grid_y + y)*grid_z + z] += grid_ptrs[t][(x*grid_y + y)*grid_z + z];
  }

#elif defined(USE_OMPT)
  auto grid_x = grid.extent(0);
  #pragma omp target teams distribute parallel for simd \
                     is_device_ptr(pos_p,grid_p,val_p) \
                     map(to: N,dx,grid_y,grid_x)
  for(int i = 0; i<N; i++) {
    int x = pos_p[i*3]/dx;
    int y = pos_p[i*3+1]/dx;
    int z = pos_p[i*3+2]/dx;
    // Matching LayoutLeft indexing Kokkos GPU variants
    #pragma omp atomic update
    grid_p[x + grid_x * (y + z * grid_y)] += val_p[i];
  }
#elif defined(USE_CUDA)
  auto grid_x = grid.extent(0);
  cuda_kernel<<<(N+128)/128,128>>>(N,[=]__device__(int i) {
    int x = pos_p[i*3]/dx;
    int y = pos_p[i*3+1]/dx;
    int z = pos_p[i*3+2]/dx;
    // Matching LayoutLeft indexing Kokkos GPU variants
    atomicAdd(&grid_p[x + grid_x * (y + z * grid_y)], val_p[i]);
  });
#endif

#endif
}

template<class T>
double run_test(int R, double dx, particle_pos_t pos, const int grid_size) {
  Kokkos::View<T*> values("Values",pos.extent(0));
  Kokkos::View<T***> grid("grid",grid_size+1,grid_size+1,grid_size+1);
  Kokkos::deep_copy(values, T(1.));
#ifdef USE_OMP
  Kokkos::View<T****> grid_sv("grid_copy",Kokkos::OpenMP().concurrency(),grid_size+1,grid_size+1,grid_size+1);
#else
  Kokkos::Experimental::ScatterView<T***> grid_sv(grid);
#endif
  Kokkos::fence();
  discretize(dx,pos,values,grid,grid_sv);

  Kokkos::fence();
  Kokkos::Timer timer;
  for(int r=0; r<R; r++)
    discretize(dx,pos,values,grid,grid_sv);
  Kokkos::fence();
  double time = timer.seconds();
  return time;
}

int main(int argc, char *argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);

  int const num_particles = argc>1?atoi(argv[1]):10000000;
  int const grid_size = argc>2?atoi(argv[2]):20;
  const int R = argc>3?atoi(argv[3]):20;
  double dx = 1./grid_size;

  particle_pos_t pos("a",num_particles);
  Kokkos::Random_XorShift64_Pool<> g(1931);
    Kokkos::fill_random(pos, g, 1.);

  Kokkos::fence();

  double time_int = run_test<int>(R, dx, pos, grid_size);
#if !defined(KOKKOS_ENABLE_HIP)
  double time_double = run_test<double>(R, dx, pos, grid_size);
#endif
#if defined(USE_KOKKOS) || defined(USE_OMP)
  double time_complex_float = run_test<Kokkos::complex<float>>(R, dx, pos, grid_size);
  double time_complex_double = run_test<Kokkos::complex<double>>(R, dx, pos, grid_size);
#endif

  double gup = 1.e-9*num_particles*R;
  printf("N: %i Grid: %i Time: %e %e %e %e GUPS: %e %e %e %e\n", num_particles, grid_size,
    time_int/R, time_double/R, time_complex_float/R, time_complex_double/R,
    gup/time_int, gup/time_double, gup/time_complex_float, gup/time_complex_double);
  return 0;
}
