#include <stdio.h>

__global__
void shuffleUp(float* out, float* in, int count) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx > count) return;

	float local_value = in[idx];
	local_value = __shfl_up_sync(0xffffffff, local_value, 1);

	out[idx] = local_value;
}

__global__
void shuffleDown(float* out, float* in, int count) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx > count) return;

	float local_value = in[idx];
	local_value = __shfl_down_sync(0xffffffff, local_value, 1);

	out[idx] = local_value;
}

__global__
void shuffleButterfly(float* out, float* in, int count) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx > count) return;

	float local_value = in[idx];
	local_value = __shfl_xor_sync(0xffffffff, local_value, 1);

	out[idx] = local_value;
}

void printArray(const char* label, float* array, int count) {
	printf("%-20s ", label);
	for (int x = 0; x < count; x++) { printf("%3d", int(array[x])); }
	printf("\n");
}

int main(void) {
	printf("\n");

	int count = 16;
	dim3 block(8);
	dim3 grid((count + block.x - 1) / block.x);

	float* h_array = (float*)malloc(count*sizeof(float));
	for (int x = 0; x < count; x++) { h_array[x] = x; }
	printArray("array: ", h_array, count);

	float* h_result_array = (float*)malloc(count*sizeof(float));

	float *d_array, *d_result_array;
	cudaMalloc((float**)&d_array, count*sizeof(float));
	cudaMemcpy(d_array, h_array, count*sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc((float**)&d_result_array, count*sizeof(float));

	shuffleUp<<<grid, block>>>(d_result_array, d_array, count);
	cudaDeviceSynchronize();
	cudaMemcpy(h_result_array, d_result_array, count*sizeof(float), cudaMemcpyDeviceToHost);
	printArray("shuffleUp: ", h_result_array, count);

	shuffleDown<<<grid, block>>>(d_result_array, d_array, count);
	cudaDeviceSynchronize();
	cudaMemcpy(h_result_array, d_result_array, count*sizeof(float), cudaMemcpyDeviceToHost);
	printArray("shuffleDown: ", h_result_array, count);

	shuffleButterfly<<<grid, block>>>(d_result_array, d_array, count);
	cudaDeviceSynchronize();
	cudaMemcpy(h_result_array, d_result_array, count*sizeof(float), cudaMemcpyDeviceToHost);
	printArray("shuffleButterfly: ", h_result_array, count);

	free(h_array);
	free(h_result_array);
	cudaFree(d_array);
	cudaFree(d_result_array);
	cudaDeviceReset();

	printf("\n");
	return 0;
}