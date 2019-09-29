//input: integer
//output: sum of all numbers in [0, integer)

#include <stdio.h>
#include "analysis.h"

__global__
void arraySum(int* partial, int* array, int count) {
	*partial += blockIdx.x * blockDim.x + threadIdx.x;
}

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
	int idx = (2 * blockIdx.x * blockDim.x) + threadIdx.x;

	if (idx + blockDim.x < count) {
		array[idx] += array[idx + blockDim.x]; 
	}
	__syncthreads();

	int* local_array = array + (2 * blockIdx.x * blockDim.x);

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
void arrayPartialSumUnrolled8(int* partial, int* array, int count) {
	int idx = (8 * blockIdx.x * blockDim.x) + threadIdx.x;
	
	if (idx + (7 * blockDim.x) < count) {
		array[idx] += array[idx + blockDim.x];
		array[idx] += array[idx + (2 * blockDim.x)];
		array[idx] += array[idx + (3 * blockDim.x)];
		array[idx] += array[idx + (4 * blockDim.x)];
		array[idx] += array[idx + (5 * blockDim.x)];
		array[idx] += array[idx + (6 * blockDim.x)];
		array[idx] += array[idx + (7 * blockDim.x)];
	}
	__syncthreads();

	int* local_array = array + (8 * blockIdx.x * blockDim.x);

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

int callArrayPartialSumUnrolled2Kernel(int count) {
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

	arrayPartialSumUnrolled2<<<grid.x / 2,block>>>(device_partial, device_array, count);
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

int callArrayPartialSumUnrolled8Kernel(int count) {
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

	arrayPartialSumUnrolled8<<<grid.x / 8,block>>>(device_partial, device_array, count);
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
	Analysis::createLabel(1, "arrayPartialSumUnrolled2");
	Analysis::createLabel(2, "arrayPartialSumUnrolled8");

	cudaDeviceReset();

	Analysis::begin();
	printf("\n%-50s %d\n", "arrayPartialSum:", callArrayPartialSumKernel(input));
	Analysis::end(0);

	Analysis::begin();
	printf("%-50s %d\n", "arrayPartialSumUnrolled2:", callArrayPartialSumUnrolled2Kernel(input));
	Analysis::end(1);

	Analysis::begin();
	printf("%-50s %d\n", "arrayPartialSumUnrolled8:", callArrayPartialSumUnrolled8Kernel(input));
	Analysis::end(2);

	Analysis::printAll();
	
	return 0;
}