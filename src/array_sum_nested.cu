//input: integer
//output: sum of all numbers in [0, integer)

#include <stdio.h>
#include "analysis.h"

__global__
void nestedHelloWorld(int count, int depth) {
	printf("Depth %d, Thread %d, Block %d: Hello World!\n", depth, threadIdx.x, blockIdx.x);

	if (count == 1) { return; }

	int threadCount = count >> 1;
	if (threadIdx.x == 0 && threadCount > 0) {
		printf("\n");
		nestedHelloWorld<<<1, threadCount>>>(threadCount, ++depth);
	}
}

__global__
void arrayPartialSumNested(int* partial, int* array, int count) {
	int* local_array = array + (blockDim.x * blockIdx.x);
	int* local_partial = &partial[blockIdx.x];

	if (count == 2 && threadIdx.x == 0) {
		local_partial[0] = local_array[0] + local_array[1];
	}

	int stride = count >> 1;
	if (stride > 1 && threadIdx.x < stride) {
		local_array[threadIdx.x] += local_array[threadIdx.x + stride];
	}

	__syncthreads();

	if (threadIdx.x == 0) {
		arrayPartialSumNested<<<1, stride>>>(local_partial, local_array, stride);
	}
}

void callNestedHelloWord() {
	int block = 8;
	int grid = 1;

	nestedHelloWorld<<<grid,block>>>(block, 0);
	cudaDeviceSynchronize();
	cudaDeviceReset();
}

int callArrayPartialSumNestedKernel(int* host_array, int count) {
	dim3 block = (64);
	dim3 grid = ((count + block.x - 1) / block.x);

	int* device_array;
	cudaMalloc((int**)&device_array, count*sizeof(int));
	cudaMemcpy(device_array, host_array, count*sizeof(int), cudaMemcpyHostToDevice);

	int* host_partial = (int*)malloc(grid.x*sizeof(int));

	int* device_partial;
	cudaMalloc((int**)&device_partial, grid.x*sizeof(int));

	Analysis::begin();
	arrayPartialSumNested<<<grid,block>>>(device_partial, device_array, block.x);
	cudaDeviceSynchronize();
	Analysis::end(0);

	cudaMemcpy(host_partial, device_partial, grid.x*sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(device_array);
	cudaFree(device_partial);

	int sum = 0;
	for (int x = 0; x < grid.x; x++) {
		sum += host_partial[x];
	}
	free(host_partial);

	cudaDeviceReset();

	return sum;
}

int main(void) {
	int input = 1 << 16;

	int* host_array = (int*)malloc(input*sizeof(int));
	for (int x = 0; x < input; x++) { host_array[x] = x; }

	Analysis::setAbsoluteStart();
	Analysis::createLabel(0, "arrayPartialSumNested");

	cudaDeviceReset();

	printf("\n%-60s %d\n", "arrayPartialSumNested:", callArrayPartialSumNestedKernel(host_array, input));

	Analysis::printAll();

	return 0;
}