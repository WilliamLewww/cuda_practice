#include <stdio.h>
#define RADIUS 4
#define BLOCK_DIM_X 32

__constant__ float coef[RADIUS + 1];

__global__
void stencil(float* out, float* in) {
	__shared__ float smem[BLOCK_DIM_X + (2 * RADIUS)];

	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	int sidx = threadIdx.x + RADIUS;
	smem[sidx] = in[idx];

	if (threadIdx.x < RADIUS) {
		smem[sidx - RADIUS] = in[idx - RADIUS];
		smem[sidx + blockDim.x] = in[idx + blockDim.x];
	}

	__syncthreads();

	float tmp = 0.0f;

	#pragma unroll
	for (int x = 0; x <= RADIUS; x++) {
		tmp += coef[x] * (smem[sidx + x] - smem[sidx - x]);
	}

	out[idx] = tmp;
}

void printArray(float* array, int count) {
	for (int x = 0; x < count; x++) {
		printf("%-3d", int(array[x]));
	}

	printf("\n");
}

int main(void) {
	printf("\n");

	int count = 25;
	dim3 block(BLOCK_DIM_X);
	dim3 grid(1);

	const float h_coef[] = { 5, 4, 3, 2, 1 };
	cudaMemcpyToSymbol(coef, h_coef, (RADIUS + 1) * sizeof(float));

	float* host_array = (float*)malloc(count*sizeof(float));
	for (int x = 0; x < count; x++) { host_array[x] = x; }
	float* host_result_array = (float*)malloc(count*sizeof(float));

	float *device_array, *device_result_array;
	cudaMalloc((float**)&device_array, count*sizeof(float));
	cudaMemcpy(device_array, host_array, count*sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc((float**)&device_result_array, count*sizeof(float));

	stencil<<<grid, block>>>(device_result_array, device_array);
	cudaDeviceSynchronize();
	cudaMemcpy(host_result_array, device_result_array, count*sizeof(float), cudaMemcpyDeviceToHost);

	printArray(host_result_array, count);

	cudaFree(device_array);
	cudaFree(device_result_array);
	cudaDeviceReset();

	printf("\n");
	return 0;
}