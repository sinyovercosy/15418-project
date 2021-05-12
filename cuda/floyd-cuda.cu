#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#define B 16
#define RC(i, j) (i * N + j)

__global__ void floyd_kernel(int* D, int N, int k) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= N || j >= N)
    return;
  int d = D[RC(i, k)] + D[RC(k, j)];
  if (d < D[RC(i, j)]) {
    D[RC(i, j)] = d;
  }
}

__device__ inline void floyd_block(int* D, int N, int i, int j, int bk) {
  for (int k = bk * B; k < bk * B + B; k++) {
    int d = D[RC(i, k)] + D[RC(k, j)];
    if (d < D[RC(i, j)]) {
      D[RC(i, j)] = d;
    }
  }
}

__global__ void floyd_block_kernel1(int* D, int N, int bk) {
  int i = bk * B + threadIdx.y;
  int j = bk * B + threadIdx.x;

  floyd_block(D, N, i, j, bk);
}

__global__ void floyd_block_kernel2(int* D, int N, int bk) {
  if (blockIdx.x == bk)
    return;
  int i = blockIdx.x * B + threadIdx.y;
  int j = bk * B + threadIdx.x;

  floyd_block(D, N, i, j, bk);
  floyd_block(D, N, j, i, bk);
}

__global__ void floyd_block_kernel3(int* D, int N, int bk) {
  if (blockIdx.x == bk || blockIdx.y == bk)
    return;
  int i = blockIdx.y * B + threadIdx.y;
  int j = blockIdx.x * B + threadIdx.x;

  floyd_block(D, N, i, j, bk);
}

void floyd_cuda(int* input, int* output, int N) {
  // compute number of blocks and threads per block
  const dim3 block_dim(B, B);
  const dim3 grid_dim((N + B - 1) / B, (N + B - 1) / B);

  int* device_data;
  cudaMalloc(&device_data, N * N * sizeof(int));

  cudaMemcpy(device_data, input, N * N * sizeof(int), cudaMemcpyHostToDevice);

  for (int bk = 0; bk < N / B; bk++) {
    floyd_block_kernel1<<<1, block_dim>>>(device_data, N, bk);
    floyd_block_kernel2<<<grid_dim.x, block_dim>>>(device_data, N, bk);
    floyd_block_kernel3<<<grid_dim, block_dim>>>(device_data, N, bk);
    cudaDeviceSynchronize();
  }

  cudaMemcpy(output, device_data, N * N * sizeof(float),
             cudaMemcpyDeviceToHost);

  cudaError_t errCode = cudaPeekAtLastError();
  if (errCode != cudaSuccess) {
    fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode,
            cudaGetErrorString(errCode));
  }

  cudaFree(device_data);
}

void printCudaInfo() {
  // for fun, just print out some stats on the machine

  int deviceCount = 0;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);

  printf("---------------------------------------------------------\n");
  printf("Found %d CUDA devices\n", deviceCount);

  for (int i = 0; i < deviceCount; i++) {
    cudaDeviceProp deviceProps;
    cudaGetDeviceProperties(&deviceProps, i);
    printf("Device %d: %s\n", i, deviceProps.name);
    printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
    printf("   Global mem: %.0f MB\n",
           static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
    printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
  }
  printf("---------------------------------------------------------\n");
}
