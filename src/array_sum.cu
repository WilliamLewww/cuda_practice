//input: integer
//output: sum of all numbers in [0, integer)

#include <stdio.h>
#include "analysis.h"

int arraySum(int* array, int count) {
	int sum = 0;
	for (int x = 0; x < count; x++) {
		sum += array[x];
	}

	return sum;
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

__global__
void arrayPartialSumUnrolledWarp8(int* partial, int* array, int count) {
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

	for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
		if (threadIdx.x < stride) {
			local_array[threadIdx.x] += local_array[threadIdx.x + stride];
		}
		__syncthreads();
	}

	if (threadIdx.x < 32) {
		volatile int* v_array = local_array;
		v_array[threadIdx.x] += v_array[threadIdx.x + 32];
		v_array[threadIdx.x] += v_array[threadIdx.x + 16];
		v_array[threadIdx.x] += v_array[threadIdx.x + 8];
		v_array[threadIdx.x] += v_array[threadIdx.x + 4];
		v_array[threadIdx.x] += v_array[threadIdx.x + 2];
		v_array[threadIdx.x] += v_array[threadIdx.x + 1];
	}

	if (threadIdx.x == 0) {
		partial[blockIdx.x] = local_array[0];
	}
}

int callArrayPartialSumKernel(int* host_array, int count) {
	dim3 block = (64);
	dim3 grid = ((count + block.x - 1) / block.x);

	int* device_array;
	cudaMalloc((int**)&device_array, count*sizeof(int));
	cudaMemcpy(device_array, host_array, count*sizeof(int), cudaMemcpyHostToDevice);

	int* host_partial = (int*)malloc(grid.x*sizeof(int));

	int* device_partial;
	cudaMalloc((int**)&device_partial, grid.x*sizeof(int));

	Analysis::begin();
	arrayPartialSum<<<grid,block>>>(device_partial, device_array, count);
	cudaDeviceSynchronize();
	Analysis::end(1);

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

int callArrayPartialSumUnrolled2Kernel(int* host_array, int count) {
	dim3 block = (64);
	dim3 grid = ((count + block.x - 1) / block.x);

	int* device_array;
	cudaMalloc((int**)&device_array, count*sizeof(int));
	cudaMemcpy(device_array, host_array, count*sizeof(int), cudaMemcpyHostToDevice);

	int* host_partial = (int*)malloc(grid.x*sizeof(int));

	int* device_partial;
	cudaMalloc((int**)&device_partial, grid.x*sizeof(int));

	Analysis::begin();
	arrayPartialSumUnrolled2<<<grid.x / 2,block>>>(device_partial, device_array, count);
	cudaDeviceSynchronize();
	Analysis::end(2);

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

int callArrayPartialSumUnrolled8Kernel(int* host_array, int count) {
	dim3 block = (64);
	dim3 grid = ((count + block.x - 1) / block.x);

	int* device_array;
	cudaMalloc((int**)&device_array, count*sizeof(int));
	cudaMemcpy(device_array, host_array, count*sizeof(int), cudaMemcpyHostToDevice);

	int* host_partial = (int*)malloc(grid.x*sizeof(int));

	int* device_partial;
	cudaMalloc((int**)&device_partial, grid.x*sizeof(int));

	Analysis::begin();
	arrayPartialSumUnrolled8<<<grid.x / 8,block>>>(device_partial, device_array, count);
	cudaDeviceSynchronize();
	Analysis::end(3);

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

int callArrayPartialSumUnrolledWarp8Kernel(int* host_array, int count) {
	dim3 block = (64);
	dim3 grid = ((count + block.x - 1) / block.x);

	int* device_array;
	cudaMalloc((int**)&device_array, count*sizeof(int));
	cudaMemcpy(device_array, host_array, count*sizeof(int), cudaMemcpyHostToDevice);

	int* host_partial = (int*)malloc(grid.x*sizeof(int));

	int* device_partial;
	cudaMalloc((int**)&device_partial, grid.x*sizeof(int));

	Analysis::begin();
	arrayPartialSumUnrolledWarp8<<<grid.x / 8,block>>>(device_partial, device_array, count);
	cudaDeviceSynchronize();
	Analysis::end(4);

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
	Analysis::createLabel(0, "arraySum (CPU)");
	Analysis::createLabel(1, "arrayPartialSum");
	Analysis::createLabel(2, "arrayPartialSumUnrolled2");
	Analysis::createLabel(3, "arrayPartialSumUnrolled8");
	Analysis::createLabel(4, "arrayPartialSumUnrolledWarp8Kernel");

	cudaDeviceReset();

	Analysis::begin();
	printf("\n%-50s %d\n", "arraySum (CPU):", arraySum(host_array, input));
	Analysis::end(0);

	printf("%-50s %d\n", "arrayPartialSum:", callArrayPartialSumKernel(host_array, input));
	printf("%-50s %d\n", "arrayPartialSumUnrolled2:", callArrayPartialSumUnrolled2Kernel(host_array, input));
	printf("%-50s %d\n", "arrayPartialSumUnrolled8:", callArrayPartialSumUnrolled8Kernel(host_array, input));
	printf("%-50s %d\n", "callArrayPartialSumUnrolledWarp8Kernel:", callArrayPartialSumUnrolledWarp8Kernel(host_array, input));

	Analysis::printAll();
	
	return 0;
}