//input: integer
//output: sum of all numbers in [0, integer]

#include <stdio.h>

__global__
void arrayPartialSum(int* partialSumArray, int* array, int count) {

}

int main(void) {
	int input = 1 << 16;

	int* h_array = (int*)malloc(input*sizeof(int));
	for (int x = 0; x < input; x++) {
		h_array[input] = x;
	}

	dim3 block = (64);
	dim3 grid = ((input + block.x - 1) / block.x);

	int* d_array;
	cudaMalloc((int**)&d_array, input*sizeof(int));
	cudaMemcpy(d_array, h_array, input*sizeof(int), cudaMemcpyHostToDevice);

	int* d_partialSumArray;
	cudaMalloc((int**)&d_partialSumArray, grid.x*sizeof(int));

	arrayPartialSum<<<grid, block>>>(d_partialSumArray, d_array, input);
	cudaDeviceSynchronize();

	cudaDeviceReset();
	return 0;
}