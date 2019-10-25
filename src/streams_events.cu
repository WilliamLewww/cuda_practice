#include <stdio.h>
#define STREAM_COUNT 3
#define ELEMENT_COUNT 300000

__global__ 
void kernel0() {
	float sum = 0.0;
	for (int x = 0; x < ELEMENT_COUNT; x++) {
		for (int y = 0; y < ELEMENT_COUNT; y++) {
			sum = sum + tan(0.1) * tan(0.1);
		}
	}
}

__global__ 
void kernel1() {
	float sum = 0.0;
	for (int x = 0; x < ELEMENT_COUNT; x++) {
		for (int y = 0; y < ELEMENT_COUNT; y++) {
			sum = sum + tan(0.1) * tan(0.1);
		}
	}
}

__global__ 
void kernel2() {
	float sum = 0.0;
	for (int x = 0; x < ELEMENT_COUNT; x++) {
		for (int y = 0; y < ELEMENT_COUNT; y++) {
			sum = sum + tan(0.1) * tan(0.1);
		}
	}
}

void warmUp(dim3 block, dim3 grid) {
	kernel0<<<grid,block>>>();
	kernel1<<<grid,block>>>();
	kernel2<<<grid,block>>>();
}

void createArraysNullStream(dim3 block, dim3 grid) {

}

void createArrayNonNullStream(dim3 block, dim3 grid) {
	cudaStream_t* streams = (cudaStream_t*)malloc(STREAM_COUNT*sizeof(cudaStream_t));
	for (int x = 0; x < STREAM_COUNT; x++) {
		cudaStreamCreate(&streams[x]);
	}

	for (int x = 0; x < STREAM_COUNT; x++) {
		kernel0<<<grid,block,0,streams[x]>>>();
		kernel1<<<grid,block,0,streams[x]>>>();
		kernel2<<<grid,block,0,streams[x]>>>();
	}

	for (int x = 0; x < STREAM_COUNT; x++) {
		cudaStreamDestroy(streams[x]);
	}
}

int main(void) {
	dim3 block(1);
	dim3 grid(1);

	warmUp(block, grid);
	createArraysNullStream(block, grid);
	createArrayNonNullStream(block, grid);
	cudaDeviceReset();
}