#include <stdio.h>

#define OUTER_LOOP_COUNT 3000
#define INNER_LOOP_COUNT 1000

__global__
void warmUp(int* out, int* in, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= size) return;
  out[idx] = in[idx];
}

__global__
void branchKernel(int* out, int* in, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= size) return;

  for (int x = 0; x < OUTER_LOOP_COUNT; x++) {
    for (int y = 0; y < INNER_LOOP_COUNT; y++) {
      if (idx % 2 == 0) {
        out[idx] = in[idx] * 2;
      }
    }
  }
}

__global__
void calculateKernel(int* out, int* in, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= size) return;

  for (int x = 0; x < OUTER_LOOP_COUNT; x++) {
    for (int y = 0; y < INNER_LOOP_COUNT; y++) {
      out[idx] = in[idx] * 2 * (idx % 2 == 0);
    }
  }
}

int main(void) {
  int dataCount = 10000;
  int* h_data = (int*)malloc(dataCount*sizeof(int));

  dim3 block(32);
  dim3 grid((block.x + dataCount - 1) / block.x);

  int *d_data, *d_result;
  cudaMalloc(&d_data, dataCount*sizeof(int));
  cudaMalloc(&d_result, dataCount*sizeof(int));
  cudaMemcpy(d_data, h_data, dataCount*sizeof(int), cudaMemcpyHostToDevice);

  warmUp<<<grid, block>>>(d_result, d_data, dataCount);
  cudaDeviceSynchronize();

  branchKernel<<<grid, block>>>(d_result, d_data, dataCount);
  cudaDeviceSynchronize();

  calculateKernel<<<grid, block>>>(d_result, d_data, dataCount);
  cudaDeviceSynchronize();

  cudaFree(d_result);
  cudaFree(d_data);
  free(h_data);
  
  return 0;
}