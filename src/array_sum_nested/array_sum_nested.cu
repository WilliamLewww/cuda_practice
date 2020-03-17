#include <stdio.h>
#include "../common_headers/analysis.h"

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
		cudaDeviceSynchronize();
	}
	__syncthreads();
}

__global__
void arrayPartialSumNestedNoSync(int* partial, int* array, int count) {
	int* local_array = array + (blockDim.x * blockIdx.x);
	int* local_partial = &partial[blockIdx.x];

	if (count == 2 && threadIdx.x == 0) {
		local_partial[0] = local_array[0] + local_array[1];
	}

	int stride = count >> 1;
	if (stride > 1 && threadIdx.x < stride) {
		local_array[threadIdx.x] += local_array[threadIdx.x + stride];
		if (threadIdx.x == 0) {
			arrayPartialSumNested<<<1, stride>>>(local_partial, local_array, stride);
		}
	}
}

__global__
void arrayPartialSumNestedLess(int* partial, int* array, int stride, int const dimension) {
	int* local_array = array + (blockIdx.x * dimension);

	if (stride == 1 && threadIdx.x == 0) {
		partial[blockIdx.x] += local_array[0] + local_array[1];
		return;
	}

	local_array[threadIdx.x] += local_array[threadIdx.x + stride];
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		arrayPartialSumNestedLess<<<gridDim.x, stride / 2>>>(partial, array, stride / 2, dimension);
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
	for (int x = 0; x < grid.x; x++) { sum += host_partial[x]; }
	free(host_partial);

	cudaDeviceReset();

	return sum;
}

int callArrayPartialSumNestedNoSyncKernel(int* host_array, int count) {
	dim3 block = (64);
	dim3 grid = ((count + block.x - 1) / block.x);

	int* device_array;
	cudaMalloc((int**)&device_array, count*sizeof(int));
	cudaMemcpy(device_array, host_array, count*sizeof(int), cudaMemcpyHostToDevice);

	int* host_partial = (int*)malloc(grid.x*sizeof(int));

	int* device_partial;
	cudaMalloc((int**)&device_partial, grid.x*sizeof(int));

	Analysis::begin();
	arrayPartialSumNestedNoSync<<<grid,block>>>(device_partial, device_array, block.x);
	cudaDeviceSynchronize();
	Analysis::end(1);

	cudaMemcpy(host_partial, device_partial, grid.x*sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(device_array);
	cudaFree(device_partial);

	int sum = 0;
	for (int x = 0; x < grid.x; x++) { sum += host_partial[x]; }
	free(host_partial);

	cudaDeviceReset();

	return sum;
}

int callArrayPartialSumNestedLessKernel(int* host_array, int count) {
	dim3 block = (64);
	dim3 grid = ((count + block.x - 1) / block.x);

	int* device_array;
	cudaMalloc((int**)&device_array, count*sizeof(int));
	cudaMemcpy(device_array, host_array, count*sizeof(int), cudaMemcpyHostToDevice);

	int* host_partial = (int*)malloc(grid.x*sizeof(int));

	int* device_partial;
	cudaMalloc((int**)&device_partial, grid.x*sizeof(int));

	Analysis::begin();
	arrayPartialSumNestedLess<<<grid,block.x / 2>>>(device_partial, device_array, block.x / 2, block.x);
	cudaDeviceSynchronize();
	Analysis::end(2);

	cudaMemcpy(host_partial, device_partial, grid.x*sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(device_array);
	cudaFree(device_partial);

	int sum = 0;
	for (int x = 0; x < grid.x; x++) { sum += host_partial[x]; }
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
	Analysis::createLabel(1, "arrayPartialSumNestedNoSync");
	Analysis::createLabel(2, "arrayPartialSumNestedLess");

	cudaDeviceReset();

	printf("\n%-60s %d\n", "arrayPartialSumNested:", callArrayPartialSumNestedKernel(host_array, input));
	printf("%-60s %d\n", "arrayPartialSumNestedNoSync:", callArrayPartialSumNestedNoSyncKernel(host_array, input));
	printf("%-60s %d\n", "arrayPartialSumNestedLess:", callArrayPartialSumNestedLessKernel(host_array, input));

	Analysis::printAll();

	return 0;
}