#include <iostream>
#include <omp.h>

#if defined(OMP_KERNEL_MODE)
#include <ompx.h>
#endif

int main (int argc, char *argv[]) {
  const int N = 10;
  const int total_elements = N*N*N;
  
  int* a = static_cast<int*> (omp_target_alloc(total_elements * sizeof(int), omp_get_initial_device()));
  int* d_a = static_cast<int*> (omp_target_alloc(total_elements * sizeof(int), omp_get_default_device()));

#if !defined(OMP_KERNEL_MODE)
  #pragma omp target teams distribute parallel for collapse(3) firstprivate(d_a)
  for(int i = 0; i < N; ++i)
    for (int j = 0; j < N; j++) {
      for(int k = 0; k < N; ++k)
        d_a[i*N*N + j*N + k] = i+1;
    }
  /*#pragma omp parallel for */
  /*for(int i_ = 0; i_ < total_elements; ++i_) {*/
  /*  const int i = i_ / (N*N);*/
  /*  const int j_ = i_ % (N*N);*/
  /*  const int j = j_ / N;*/
  /*  const int k = j_ % N;*/
  /*  a[i*N*N + j*N + k] = i+1;*/
  /*}*/

  #else
  const int team_size=32;
  const int num_teams = (total_elements%team_size) ? total_elements/team_size +1: total_elements/team_size; 
  printf("num_teams = %d\n",num_teams);
#pragma omp target teams ompx_bare \
  num_teams(num_teams, 1, 1) thread_limit(team_size, 1, 1) \
  firstprivate(d_a) 
  {
    const int blockIdx  = ompx::block_id(ompx::dim_x);
    const int blockDimx = ompx::block_dim(ompx::dim_x);
    const int threadIdx = ompx::thread_id(ompx::dim_x);
    const int index = blockIdx * blockDimx + threadIdx;

    if(index < total_elements)
    {
      const int i = index / (N*N);
      const int j_ = index % (N*N);
      const int j = j_ / N;
      const int k = j_ % N;
      d_a[i*N*N + j*N + k] = i+1;
    }
  }

#endif
  omp_target_memcpy(a, d_a, total_elements * sizeof(int), 0, 0, omp_get_initial_device(), omp_get_default_device());

// Validate results.
  int result = 0;
  #pragma omp parallel for reduction(+:result) collapse(3)
  for(int i = 0; i < N; ++i)
    for (int j = 0; j < N; j++) {
      for(int k = 0; k < N; ++k)
        result += a[i*N*N + j*N + k];
    }

  const int expected = N*N*N*(N+1)/2;
  if(result == expected)
    printf("success: result = %d, expected = %d\n",result,expected);
  else
    printf("failure: result = %d, expected = %d\n",result,expected);
  
  return 0;
}
