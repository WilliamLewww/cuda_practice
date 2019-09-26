//input: integer
//output: sum of all numbers in [0, integer)

#include <stdio.h>

__global__
void arrayPartialSum(int* array, int count) {
	if (blockIdx.x * blockDim.x + threadIdx.x >= count) return;

	int* local_array = array + (blockIdx.x * blockDim.x);

	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (threadIdx.x < stride) {
			local_array[threadIdx.x] += local_array[threadIdx.x + stride];
		}

		__syncthreads();
	}

	if (threadIdx.x == 0) {
		printf("%d\n", local_array[0]);
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

	arrayPartialSum<<<grid,block>>>(device_array, input);

	cudaDeviceSynchronize();
	cudaDeviceReset();
	return 0;
}