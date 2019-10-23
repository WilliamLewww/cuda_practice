#include <stdio.h>
#define STREAM_COUNT 3

__global__
void createIncrementingArray(float* out, int count) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (idx < count) {
		out[idx] = idx;
	}
}

void warmUp(int count, dim3 block, dim3 grid) {
	float* result;
	cudaMalloc((float**)&result, count*sizeof(float));

	createIncrementingArray<<<grid, block>>>(result, count);
	cudaFree(result);
}

void createArraysNullStream(int count, dim3 block, dim3 grid) {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	float *d_result_first, *d_result_second, *d_result_third;
	cudaMalloc((float**)&d_result_first, count*sizeof(float));
	cudaMalloc((float**)&d_result_second, count*sizeof(float));
	cudaMalloc((float**)&d_result_third, count*sizeof(float));

	for (int x = 0; x < STREAM_COUNT; x++) {
		createIncrementingArray<<<grid, block>>>(d_result_first, count);
		createIncrementingArray<<<grid, block>>>(d_result_second, count);
		createIncrementingArray<<<grid, block>>>(d_result_third, count);
	}

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float time;
	cudaEventElapsedTime(&time, start, stop);

	printf("%-30s%f\n", "Null Stream: ", time);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree(d_result_first);
	cudaFree(d_result_second);
	cudaFree(d_result_third);
}

void createArrayNonNullStream(int count, dim3 block, dim3 grid) {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	float *d_result_first, *d_result_second, *d_result_third;
	cudaMalloc((float**)&d_result_first, count*sizeof(float));
	cudaMalloc((float**)&d_result_second, count*sizeof(float));
	cudaMalloc((float**)&d_result_third, count*sizeof(float));

	int streamCount = STREAM_COUNT;
	cudaStream_t* streams = (cudaStream_t*)malloc(streamCount*sizeof(cudaStream_t));
	for (int x = 0; x < streamCount; x++) {
		cudaStreamCreate(&streams[x]);
	}

	for (int x = 0; x < streamCount; x++) {
		createIncrementingArray<<<grid, block, 0, streams[x]>>>(d_result_first, count);
		createIncrementingArray<<<grid, block, 0, streams[x]>>>(d_result_first, count);
		createIncrementingArray<<<grid, block, 0, streams[x]>>>(d_result_first, count);
	}

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float time;
	cudaEventElapsedTime(&time, start, stop);

	printf("%-30s%f\n", "Non-Null Stream: ", time);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree(d_result_first);
	cudaFree(d_result_second);
	cudaFree(d_result_third);
}

int main(void) {
	printf("\n");
	int count = 1 << 16;
	dim3 block(32);
	dim3 grid((count + block.x - 1) / block.x);

	warmUp(count, block, grid);
	createArraysNullStream(count, block, grid);
	createArrayNonNullStream(count, block, grid);
	cudaDeviceReset();

	printf("\n");
}