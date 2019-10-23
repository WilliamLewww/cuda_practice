#include <stdio.h>

__global__
void createIncrementingArray(float* out, int count) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (idx < count) {
		out[idx] = idx;
	}
}

void createArraysNullStream(int count, dim3 block, dim3 grid) {
	float *d_result_first, *d_result_second, *d_result_third;
	cudaMalloc((float**)&d_result_first, count*sizeof(float));
	cudaMalloc((float**)&d_result_second, count*sizeof(float));
	cudaMalloc((float**)&d_result_third, count*sizeof(float));

	createIncrementingArray<<<grid, block>>>(d_result_first, count);
	createIncrementingArray<<<grid, block>>>(d_result_second, count);
	createIncrementingArray<<<grid, block>>>(d_result_third, count);
}

int main(void) {
	int count = 1 << 16;
	dim3 block(32);
	dim3 grid((count + block.x - 1) / block.x);

	createArraysNullStream(count, block, grid);

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	cudaStreamDestroy(stream);
	cudaDeviceReset();
}