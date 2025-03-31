#include <iostream>

// Define this var at compile time instead of changing in the source everytime.
// #define CUDA_MODE

#if defined(CUDA_MODE)
#include <helper_cuda.h>
#define device_fn __device__
#define sync_threads __syncthreads()
#else
#include <omp.h>
#include <ompx.h>
#define device_fn
#define sync_threads ompx_sync_block_acq_rel()
#endif

device_fn void vector_kernel(int *a, int j, int i, int N) {
  int update = 0;
#if defined(NO_SCRATCH)
  for (int k = 0; k < N; k++) {
    update += a[i * N + j];
  }
  a[i * N + j] = update;
#else
#if defined(CUDA_MODE)
  const int blockDim_x = blockDim.x;
  const int threadId_x = threadIdx.x;
  const int threadId_y = threadIdx.y;
  extern __shared__ int buf[];
#else
  const int blockDim_x = ompx::block_dim(ompx::dim_x);
  const int threadId_x = ompx::thread_id(ompx::dim_x);
  const int threadId_y = ompx::thread_id(ompx::dim_y);
  int *buf = static_cast<int *>(llvm_omp_target_dynamic_shared_alloc());
#endif
  buf[threadId_y * blockDim_x + threadId_x] = 0;
  sync_threads;
  for (int k = threadId_x; k < N; k += blockDim_x) {
    buf[threadId_y * blockDim_x + threadId_x] += a[i * N + j];
  }
  sync_threads;

  if (threadId_x == 0) {
    for (int tid = 0; tid < blockDim_x; ++tid) update += buf[tid];

    a[i * N + j] = update;
  }
  sync_threads;
#endif
}

#if defined(CUDA_MODE)
__global__ void cu_kernel(int *a, int N) {
  const int blockId_x  = blockIdx.x;
  const int blockDim_y = blockDim.y;
  const int threadId_y = threadIdx.y;

  const int i = blockId_x;
  for (int j = threadId_y; j < N; j += blockDim_y) {
    vector_kernel(a, j, i, N);
  }
}
#else
void omp_kernel(int *a, int N) {
  const int blockIdx  = ompx::block_id(ompx::dim_x);
  const int blockDimy = ompx::block_dim(ompx::dim_y);
  const int threadIdy = ompx::thread_id(ompx::dim_y);

  const int i = blockIdx;
  for (int j = threadIdy; j < N; j += blockDimy) {
    vector_kernel(a, j, i, N);
  }
}
#endif

int main(int argc, char **argv) {
  const int N           = 10;
  const int team_size   = 32;
  const int vector_size = 8;

  int *h_a = new int[N * N];
  for (int i = 0; i < N * N; ++i) h_a[i] = 1;

#if defined(CUDA_MODE)
  printf("IN cuda mode\n");
  int *a;
  checkCudaErrors(cudaMalloc(&a, N * N * sizeof(int)));
  checkCudaErrors(
      cudaMemcpy(a, h_a, N * N * sizeof(int), cudaMemcpyHostToHost));

  cu_kernel<<<dim3(N, 1, 1), dim3(vector_size, team_size, 1),
              vector_size * team_size * sizeof(int)>>>(a, N);
  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(
      cudaMemcpy(h_a, a, N * N * sizeof(int), cudaMemcpyDeviceToHost));

  cudaFree(a);
#else
  printf("IN OpenMP mode\n");
  int *a = static_cast<int *>(
      omp_target_alloc(N * N * sizeof(int), omp_get_default_device()));
  omp_target_memcpy(a, h_a, N * N * sizeof(int), 0, 0, omp_get_default_device(),
                    omp_get_initial_device());

#pragma omp target teams ompx_bare num_teams(N, 1, 1)       \
    thread_limit(vector_size, team_size, 1) firstprivate(a) \
    ompx_dyn_cgroup_mem(vector_size *team_size * sizeof(int))
  { omp_kernel(a, N); }

  omp_target_memcpy(h_a, a, N * N * sizeof(int), 0, 0, omp_get_initial_device(),
                    omp_get_default_device());

  omp_target_free(a, omp_get_default_device());
#endif
  int sum = 0;
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) sum += h_a[i * N + j];

  printf("sum = %d\n", sum);

  delete[] h_a;
  return 0;
}
