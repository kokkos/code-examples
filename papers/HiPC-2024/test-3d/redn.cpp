#include <iostream>
#include <omp.h>
#include <Kokkos_Core.hpp>
#if defined(OMP_KERNEL_MODE)
#include <ompx.h>
#endif

int main (int argc, char *argv[]) {
  const int N = 10;
  int result = 0;

#if !defined(KOKKOS_MODE)
  const int total_elements = N*N*N;

  int* a = static_cast<int*> (omp_target_alloc(total_elements * sizeof(int), omp_get_initial_device()));
  int* d_a = static_cast<int*> (omp_target_alloc(total_elements * sizeof(int), omp_get_default_device()));

  for(int i = 0; i < 10; ++i)
    a[i] = i+1;

#if !defined(OMP_KERNEL_MODE)
  printf("Native reduction\n");
  #pragma omp target teams distribute parallel for reduction(+:result)
  for(int i = 0; i < N; ++i)
    result += a[i];
#else
  printf("Native Kernel mode reduction\n");
  const int team_size=32;
  int nteams = total_elements / team_size + !!(total_elements % team_size);
  const int scratch_size = team_size*sizeof(int);
  int* block_redn_buf = static_cast<int*>(omp_target_alloc((nteams+1)*sizeof(int), omp_get_default_device()));
  omp_target_memcpy(d_a, a, N*sizeof(int),0,0,omp_get_default_device(), omp_get_initial_device());

#pragma omp target teams ompx_bare \
  num_teams(nteams, 1, 1) thread_limit(team_size, 1, 1) \
  firstprivate(d_a, block_redn_buf) 
  {
    const int blockIdx  = ompx::block_id(ompx::dim_x);
    const int blockDimx = ompx::block_dim(ompx::dim_x);
    const int tid = ompx::thread_id(ompx::dim_x);

    int* buf = static_cast<int*>(llvm_omp_target_dynamic_shared_alloc());
    buf[tid] = 0;
    auto i = tid + blockIdx * blockDimx;

    if(i < total_elements)
      buf[tid] += d_a[i];
    ompx_sync_block_acq_rel();

    if(tid == 0)
    {
      for(int j = 0; j < blockDimx; ++j)
        block_redn_buf[blockIdx] += buf[j];
    }
    ompx_sync_block_acq_rel();

    if(blockIdx == 0 && tid == 0)
          for(int j = 0; j < nteams; ++j)
            block_redn_buf[nteams] += block_redn_buf[j];
  }
  omp_target_memcpy((void*)&result, block_redn_buf+nteams, sizeof(int), 0, 0, omp_get_initial_device(), omp_get_default_device());
#endif
#else

  Kokkos::initialize();
  {
    printf("Kokkos Mode\n");
    Kokkos::View<int*,Kokkos::DefaultHostExecutionSpace> k_a("k_a",N);

    for(int i = 0; i < N; ++i)
      k_a(i) = i+1;

    auto k_d_a = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(),k_a);
    Kokkos::parallel_reduce("reduce", N, KOKKOS_LAMBDA(const int i, int &lsum){
      lsum += k_d_a(i);
    },result);
    printf("result = %d\n",result);
  }
#endif

// Validate results.
  const int expected = N*(N+1)/2;
  if(result == expected)
    printf("success: result = %d, expected = %d\n",result,expected);
  else
    printf("failure: result = %d, expected = %d\n",result,expected);
  
  return 0;
}
