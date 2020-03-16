#include <stdio.h>
#define ARRAY_COUNT 10

__shared__ float file_shared_array_static[ARRAY_COUNT];
extern __shared__ float file_shared_array_dynamic[];

__global__
void generateArrayStatic(float* out) {
	__shared__ float array[ARRAY_COUNT];

	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (idx < ARRAY_COUNT) {
		array[idx] = idx;
		out[idx] = array[idx];
	}
}

__global__
void generateArrayDynamic(float* out) {
	extern __shared__ float array[];

	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (idx < ARRAY_COUNT) {
		array[idx] = idx;
		out[idx] = array[idx];
	}
}

__global__
void generateArrayFileStatic(float* out) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (idx < ARRAY_COUNT) {
		file_shared_array_static[idx] = idx;
		out[idx] = file_shared_array_static[idx];
	}
}

__global__
void generateArrayFileDynamic(float* out) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (idx < ARRAY_COUNT) {
		file_shared_array_dynamic[idx] = idx;
		out[idx] = file_shared_array_dynamic[idx];
	}
}

void printArray(float* array) {
	for (int x = 0; x < ARRAY_COUNT; x++) {
		printf("%d ", int(array[x]));
	}
	printf("\n");
}

int main(void) {
	printf("\n");
	dim3 block(32);
	dim3 grid((ARRAY_COUNT+block.x-1)/block.x);

	float* host_array = (float*)malloc(ARRAY_COUNT*sizeof(float));

	float* device_array;
	cudaMalloc((float**)&device_array, ARRAY_COUNT*sizeof(float));

	generateArrayStatic<<<grid,block>>>(device_array);
	cudaDeviceSynchronize();
	cudaMemcpy(host_array, device_array, ARRAY_COUNT*sizeof(float), cudaMemcpyDeviceToHost);
	printf("%-30s", "generateArrayStatic: ");
	printArray(host_array);

	generateArrayDynamic<<<grid,block,ARRAY_COUNT*sizeof(float)>>>(device_array);
	cudaDeviceSynchronize();
	cudaMemcpy(host_array, device_array, ARRAY_COUNT*sizeof(float), cudaMemcpyDeviceToHost);
	printf("%-30s", "generateArrayDynamic: ");
	printArray(host_array);

	generateArrayFileStatic<<<grid,block>>>(device_array);
	cudaDeviceSynchronize();
	cudaMemcpy(host_array, device_array, ARRAY_COUNT*sizeof(float), cudaMemcpyDeviceToHost);
	printf("%-30s", "generateArrayFileStatic: ");
	printArray(host_array);

	generateArrayFileDynamic<<<grid,block,ARRAY_COUNT*sizeof(float)>>>(device_array);
	cudaDeviceSynchronize();
	cudaMemcpy(host_array, device_array, ARRAY_COUNT*sizeof(float), cudaMemcpyDeviceToHost);
	printf("%-30s", "generateArrayFileDynamic: ");
	printArray(host_array);

	free(host_array);
	cudaFree(device_array);
	cudaDeviceReset();
	printf("\n");
	return 0;
}