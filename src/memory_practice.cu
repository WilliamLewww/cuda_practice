#include <stdio.h>

__device__ float globalData;

__global__ 
void changeGlobalData() {
	globalData += 1.00f;
}

int main(void) {
	int device = 0;
	cudaSetDevice(device);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, device);

	printf("\nDevice #%d, %s\n", device, deviceProp.name);

	float* value = (float*)malloc(sizeof(float));
	*value = 1.23;

	cudaMemcpyToSymbol(globalData, value, sizeof(float));
	changeGlobalData<<<1,1>>>();
	cudaMemcpyFromSymbol(value, globalData, sizeof(float));

	cudaDeviceReset();

	printf("%f\n", *value);

	return 0;
}