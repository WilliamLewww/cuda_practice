#include <stdio.h>

__global__
void sumUnrollGlobal(float* out, float* in, int count) {
	float* local_array = in + (blockIdx.x * blockDim.x);

	if (blockDim.x >= 1024 && threadIdx.x < 512) { local_array[threadIdx.x] += local_array[threadIdx.x + 512]; }
	__syncthreads();
	if (blockDim.x >= 512 && threadIdx.x < 256) { local_array[threadIdx.x] += local_array[threadIdx.x + 256]; }
	__syncthreads();
	if (blockDim.x >= 256 && threadIdx.x < 128) { local_array[threadIdx.x] += local_array[threadIdx.x + 128]; }
	__syncthreads();
	if (blockDim.x >= 128 && threadIdx.x < 64) { local_array[threadIdx.x] += local_array[threadIdx.x + 64]; }
	__syncthreads();

	if (threadIdx.x < 32) {
		volatile float* v_local_array = local_array;
		v_local_array[threadIdx.x] += v_local_array[threadIdx.x + 32];
		v_local_array[threadIdx.x] += v_local_array[threadIdx.x + 16];
		v_local_array[threadIdx.x] += v_local_array[threadIdx.x + 8];
		v_local_array[threadIdx.x] += v_local_array[threadIdx.x + 4];
		v_local_array[threadIdx.x] += v_local_array[threadIdx.x + 2];
		v_local_array[threadIdx.x] += v_local_array[threadIdx.x + 1];
	}

	if (threadIdx.x == 0) { out[blockIdx.x] = local_array[0]; }
}

int main(void) {
	int count = 1 << 16;
	dim3 block(32);
	dim3 grid((count + block.x - 1) / block.x);

	float* host_array = (float*)malloc(count*sizeof(float));
	for (int x = 0; x < count; x++) { host_array[x] = x; }
	float* host_result_array = (float*)malloc(grid.x*sizeof(float));

	float *device_array, *device_result_array;
	cudaMalloc((float**)&device_array, count*sizeof(float));
	cudaMemcpy(device_array, host_array, count*sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc((float**)&device_result_array, grid.x*sizeof(float));

	sumUnrollGlobal<<<grid,block>>>(device_result_array, device_array, count);
	cudaDeviceSynchronize();

	cudaMemcpy(host_result_array, device_result_array, grid.x*sizeof(float), cudaMemcpyDeviceToHost);

	free(host_array);
	free(host_result_array);
	cudaFree(device_array);
	cudaFree(device_result_array);
	cudaDeviceReset();
	return 0;
}