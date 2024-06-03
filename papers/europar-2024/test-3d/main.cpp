#include <iostream>
#include <omp.h>
#include <ompx.h>

int main(int argc, char** argv)
{

  const int N = 10;
  const int team_size = 32;
  const int vector_size=2;

  int *a = static_cast<int*>(omp_target_alloc( N*N*sizeof(int), omp_get_default_device()));
  int *h_a = static_cast<int*>(omp_target_alloc( N*N*sizeof(int), omp_get_initial_device()));


#pragma omp target teams ompx_bare num_teams(N,1,1) thread_limit(vector_size,team_size,1) firstprivate(a)\
    ompx_dyn_cgroup_mem(vector_size * team_size * sizeof(int))
  {
      const int blockIdx  = ompx::block_id(ompx::dim_x);
      const int blockDimx = ompx::block_dim(ompx::dim_x);
      const int blockDimy = ompx::block_dim(ompx::dim_y);
      const int threadIdx = ompx::thread_id(ompx::dim_x);
      const int threadIdy = ompx::thread_id(ompx::dim_y);

      double *buf =
          static_cast<double *>(llvm_omp_target_dynamic_shared_alloc());

    buf[threadIdy * blockDimx + threadIdx] = 0.;
    ompx_sync_block_acq_rel();

    const int i = blockIdx;
    for (int j = threadIdy; j < N; j+=blockDimy)
    {
      int update=0;
      for (int k = threadIdx; k < N; k+=blockDimx)
      {
        buf[threadIdy * blockDimx + threadIdx] += i*(k+1);
//        update += i*(k+1);

      }
    ompx_sync_block_acq_rel();

      if(threadIdx == 0)
      {
        for(int tid = 0; tid < blockDimx; ++tid)
          update += buf[tid];

        a[i*N + j] = update;
      }
    }
  }

  omp_target_memcpy(h_a, a, N*N*sizeof(int), 0, 0, omp_get_initial_device(), omp_get_default_device());

  int sum = 0;
  for(int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      sum += h_a[i*N + j];

  printf("sum = %d\n", sum);
//      printf("h_a(%d,%d) = %d\n", i,j, h_a[i*N+j]);

  return 0;
}
