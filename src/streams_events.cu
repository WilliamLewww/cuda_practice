#include <stdio.h>

__global__
void createIncrementingArray(float* out, int count) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (idx < count) {
		out[idx] = idx;
	}
}

void createArraysNullStream(int count, dim3 block, dim3 grid) {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	float *d_result_first, *d_result_second, *d_result_third;
	cudaMalloc((float**)&d_result_first, count*sizeof(float));
	cudaMalloc((float**)&d_result_second, count*sizeof(float));
	cudaMalloc((float**)&d_result_third, count*sizeof(float));

	createIncrementingArray<<<grid, block>>>(d_result_first, count);
	createIncrementingArray<<<grid, block>>>(d_result_second, count);
	createIncrementingArray<<<grid, block>>>(d_result_third, count);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float time;
	cudaEventElapsedTime(&time, start, stop);

	printf("%s%f\n", "Null-Stream: ", time);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

void createArrayNonNullStream(int count, dim3 block, dim3 grid) {

}

int main(void) {
	printf("\n");
	int count = 1 << 16;
	dim3 block(32);
	dim3 grid((count + block.x - 1) / block.x);

	createArraysNullStream(count, block, grid);
	createArrayNonNullStream(count, block, grid);
	cudaDeviceReset();
	
	printf("\n");
}