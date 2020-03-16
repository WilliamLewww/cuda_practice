#include <stdio.h>
#include "analysis.h"

__device__ float globalData;

__global__ 
void changeGlobalData() {
	globalData += 1.00f;
}

__global__
void updateArray(int* array, int count) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (idx < count) {
		array[idx] *= 2.00f;
	}
}

__global__
void printDeviceMemoryArray(int* array, int count) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (idx < count) {
		printf("%d ", array[idx]);
	}
}

extern __shared__ int sharedData[];
__global__
void createArrayInSharedMemory(int count) {
    int* data = (int*)sharedData;

    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx < count) {
		data[idx] = idx;
	}
	__syncthreads();

	if (idx == 0) {
		for (int x = 0; x < count; x++) {
			printf("%d ", data[x]);
		}
		printf("\n");
	}
}

__global__
void createArrayInSharedMemoryStatic() {
	__shared__ int data[5];

    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx < 5) {
		data[idx] = idx;
	}
	__syncthreads();

	if (idx == 0) {
		for (int x = 0; x < 5; x++) {
			printf("%d ", data[x]);
		}
		printf("\n");
	}
}

void printArray(int* array, int count) {
	for (int x = 0; x < count; x++) {
		printf("%d ", array[x]);
	}
	printf("\n");
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

	// static global memory
	float* value = (float*)malloc(sizeof(float));
	*value = 1.23;

	cudaMemcpyToSymbol(globalData, value, sizeof(float));
	changeGlobalData<<<1,1>>>();
	cudaMemcpyFromSymbol(value, globalData, sizeof(float));

	printf("static global memory: %f\n", *value);

	// dynamic global memory
	int* host_memory = (int*)malloc(count*sizeof(int));
	for (int x = 0; x < count; x++) {
		host_memory[x] = x * x + 1;
	}

	int* device_memory;
	cudaMalloc((int**)&device_memory, count*sizeof(int));
	cudaMemcpy(device_memory, host_memory, count*sizeof(int), cudaMemcpyHostToDevice);
	printf("dynamic global memory: ");
	printDeviceMemoryArray<<<grid,block>>>(device_memory, count);
	cudaDeviceSynchronize();
	printf("\n");
	cudaFree(device_memory);
	free(host_memory);

	// pinned memory
	int* pinned_array;
	cudaHostAlloc((int**)&pinned_array, count* sizeof(int), cudaHostAllocDefault);

	for (int x = 0; x < count; x++) {
		pinned_array[x] = x;
	}

	updateArray<<<grid,block>>>(pinned_array, count);
	cudaDeviceSynchronize();

	printf("pinned memory: ");
	printArray(pinned_array, count);

	cudaFreeHost(pinned_array);

	// zero-copy memory (with UVA)
	int* zero_copy_array;
	cudaHostAlloc((int**)&zero_copy_array, count*sizeof(int), cudaHostAllocMapped);
	for (int x = 0; x < count; x++) {
		zero_copy_array[x] = count - x;
	}
	updateArray<<<grid,block>>>(zero_copy_array, count);
	cudaDeviceSynchronize();

	printf("zero-copy memory: ");
	printArray(zero_copy_array, count);
	cudaFreeHost(zero_copy_array);

	// managed memory
	int* managed_memory;
	cudaMallocManaged((int**)&managed_memory, count*sizeof(int));
	for (int x = 0; x < count; x++) {
		managed_memory[x] = x * 2;
	}
	updateArray<<<grid,block>>>(managed_memory, count);
	cudaDeviceSynchronize();

	printf("managed memory: ");
	printArray(managed_memory, count);
	cudaFree(managed_memory);

	// static shared memory
	printf("static shared memory: ");
	createArrayInSharedMemoryStatic<<<grid,block>>>();
	cudaDeviceSynchronize();

	// dynamic shared memory
	int sharedMemorySize = count*sizeof(float);
	printf("dynamic shared memory: ");
	createArrayInSharedMemory<<<grid,block,sharedMemorySize>>>(count);
	cudaDeviceSynchronize();

	cudaDeviceReset();
	return 0;
}