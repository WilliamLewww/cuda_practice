#include <stdio.h>
#include "analysis.h"

__device__ float globalData;

__global__ 
void changeGlobalData() {
	globalData += 1.00f;
}

__global__
void updateArray(int* array) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	array[idx] *= 2.00f;
}

int main(void) {
	int count = 5;
	int block = 64;
	int grid = (count + block - 1) / block;

	// device Information
	int device = 0;
	cudaSetDevice(device);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, device);

	printf("\ndevice #%d, %s\n", device, deviceProp.name);

	// global memory
	float* value = (float*)malloc(sizeof(float));
	*value = 1.23;

	cudaMemcpyToSymbol(globalData, value, sizeof(float));
	changeGlobalData<<<1,1>>>();
	cudaMemcpyFromSymbol(value, globalData, sizeof(float));

	printf("static global memory test: %f\n", *value);

	// pinned memory
	int* pinned_array;
	cudaHostAlloc((int**)&pinned_array, count* sizeof(int), cudaHostAllocDefault);

	for (int x = 0; x < count; x++) {
		pinned_array[x] = x;
	}

	updateArray<<<grid,block>>>(pinned_array);
	cudaDeviceSynchronize();

	printf("pinned memory test: ");
	for (int x = 0; x < count; x++) {
		printf("%d ", pinned_array[x]);
	}
	printf("\n");

	cudaFreeHost(pinned_array);

	// zero-copy memory
	int* zero_copy_array;
	cudaHostAlloc((int**)&zero_copy_array, count*sizeof(int), cudaHostAllocMapped);
	for (int x = 0; x < count; x++) {
		zero_copy_array[x] = count - x;
	}
	updateArray<<<grid,block>>>(zero_copy_array);
	cudaDeviceSynchronize();

	printf("zero-copy memory test: ");
	for (int x = 0; x < count; x++) {
		printf("%d ", zero_copy_array[x]);
	}
	printf("\n");
	cudaFreeHost(zero_copy_array);

	cudaDeviceReset();
	return 0;
}