#include <stdio.h>
#include "analysis.h"

__device__ float globalData;

__global__ 
void changeGlobalData() {
	globalData += 1.00f;
}

__global__
void updatePinnedArray(int* pinned_array) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	pinned_array[idx] *= 2.00f;
}

int main(void) {
	// device Information
	int device = 0;
	cudaSetDevice(device);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, device);

	printf("\nDevice #%d, %s\n", device, deviceProp.name);

	// global memory
	float* value = (float*)malloc(sizeof(float));
	*value = 1.23;

	cudaMemcpyToSymbol(globalData, value, sizeof(float));
	changeGlobalData<<<1,1>>>();
	cudaMemcpyFromSymbol(value, globalData, sizeof(float));

	printf("%f\n", *value);

	// pinned memory
	int count = 5;
	int block = 64;
	int grid = (count + block - 1) / block;

	int* pinned_array;
	cudaMallocHost((int**)&pinned_array, count* sizeof(int));

	for (int x = 0; x < count; x++) {
		pinned_array[x] = x;
	}

	updatePinnedArray<<<grid,block>>>(pinned_array);
	cudaDeviceSynchronize();

	for (int x = 0; x < count; x++) {
		printf("%d ", pinned_array[x]);
	}
	printf("\n");

	cudaFreeHost(pinned_array);

	cudaDeviceReset();
	return 0;
}