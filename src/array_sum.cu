//input: integer
//output: sum of all numbers in [0, integer)

#include <stdio.h>
#include "analysis.h"

__global__
void arrayPartialSum(int* partial, int* array, int count) {
	if (blockIdx.x * blockDim.x + threadIdx.x >= count) return;

	int* local_array = array + (blockIdx.x * blockDim.x);

	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (threadIdx.x < stride) {
			local_array[threadIdx.x] += local_array[threadIdx.x + stride];
		}

		__syncthreads();
	}

	if (threadIdx.x == 0) {
		partial[blockIdx.x] = local_array[0];
	}
}

__global__
void arrayPartialSumUnrolled2(int* partial, int* array, int count) {
	if (blockIdx.x * blockDim.x + threadIdx.x >= count) return;

	int* local_array = array + (blockIdx.x * blockDim.x);

	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (threadIdx.x < stride) {
			local_array[threadIdx.x] += local_array[threadIdx.x + stride];
		}

		__syncthreads();
	}

	if (threadIdx.x == 0) {
		partial[blockIdx.x] = local_array[0];
	}
}

int callArrayPartialSumKernel(int count) {
	dim3 block = (64);
	dim3 grid = ((count + block.x - 1) / block.x);

	int* host_array = (int*)malloc(count*sizeof(int));
	for (int x = 0; x < count; x++) { host_array[x] = x; }

	int* device_array;
	cudaMalloc((int**)&device_array, count*sizeof(int));
	cudaMemcpy(device_array, host_array, count*sizeof(int), cudaMemcpyHostToDevice);

	int* host_partial = (int*)malloc(grid.x*sizeof(int));

	int* device_partial;
	cudaMalloc((int**)&device_partial, grid.x*sizeof(int));

	arrayPartialSum<<<grid,block>>>(device_partial, device_array, count);
	cudaDeviceSynchronize();

	cudaMemcpy(host_partial, device_partial, grid.x*sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(device_array);
	cudaFree(device_partial);

	int sum = 0;
	for (int x = 0; x < grid.x; x++) {
		sum += host_partial[x];
	}
	free(host_array);
	free(host_partial);

	cudaDeviceReset();

	return sum;
}

int main(void) {
	int input = 1 << 16;

	Analysis::setAbsoluteStart();
	Analysis::createLabel(0, "arrayPartialSum");

	Analysis::begin();
	printf("\ntotal sum in [0, %d): %d\n", input, callArrayPartialSumKernel(input));
	Analysis::end(0);

	Analysis::printAll();
	
	return 0;
}