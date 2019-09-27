//input: integer
//output: sum of all numbers in [0, integer)

#include <stdio.h>

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

int main(void) {
	int input = 1 << 9;
	dim3 block = (32);
	dim3 grid = ((input + block.x - 1) / block.x);

	int* host_array = (int*)malloc(input*sizeof(int));
	for (int x = 0; x < input; x++) { host_array[x] = x; }

	int* device_array;
	cudaMalloc((int**)&device_array, input*sizeof(int));
	cudaMemcpy(device_array, host_array, input*sizeof(int), cudaMemcpyHostToDevice);

	int* host_partial = (int*)malloc(grid.x*sizeof(int));

	int* device_partial;
	cudaMalloc((int**)&device_partial, grid.x*sizeof(int));

	arrayPartialSum<<<grid,block>>>(device_partial, device_array, input);
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

	printf("\ntotal sum in [0, %d): %d\n\n", input, sum);

	cudaDeviceReset();
	return 0;
}