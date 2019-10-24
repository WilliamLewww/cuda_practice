#include <stdio.h>

__global__
void createIncrementingArray() {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
}

void warmUp(dim3 block, dim3 grid) {

}

void createArraysNullStream(dim3 block, dim3 grid) {

}

void createArrayNonNullStream(dim3 block, dim3 grid) {

}

int main(void) {
	printf("\n");
	dim3 block(1);
	dim3 grid(1);

	warmUp(block, grid);
	createArraysNullStream(block, grid);
	createArrayNonNullStream(block, grid);
	cudaDeviceReset();

	printf("\n");
}