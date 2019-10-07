#include <stdio.h>

void misalignedAccessHost(float* arrayA, float* arrayB, float* arrayC, int count, int offset) {
	for (int offsetIdx = offset, idx = 0; offsetIdx < count; offsetIdx++, idx++) {
		arrayC[idx] = arrayA[offsetIdx] + arrayB[offsetIdx];
	}
}

__global__
void misalignedAccessDevice(float* arrayA, float* arrayB, float* arrayC, int count, int offset) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	int offsetIdx = idx + offset;

	if (offsetIdx < count) {
		arrayC[idx] = arrayA[offsetIdx] + arrayB[offsetIdx];
	}
}

int main(void) {
	printf("\n");
	
	int count = 1 << 16;
	dim3 block = (512);
	dim3 grid = ((count + block.x - 1) / block.x);

	int offset = 5;

	float* h_a = (float*)malloc(count*sizeof(float));
	float* h_b = (float*)malloc(count*sizeof(float));

	for (int x = 0; x < count; x++) {
		h_a[x] = x * 2;
		h_b[x] = (x * 2) + 1;
	}

	float* h_result = (float*)malloc(count*sizeof(float));
	float* d_result = (float*)malloc(count*sizeof(float));

	misalignedAccessHost(h_a, h_b, h_result, count, offset);

	float *d_a, *d_b, *d_c;
	cudaMalloc((float**)&d_a, count*sizeof(float));
	cudaMemcpy(d_a, h_a, count*sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc((float**)&d_b, count*sizeof(float));
	cudaMemcpy(d_b, h_b, count*sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc((float**)&d_c, count*sizeof(float));

	misalignedAccessDevice<<<grid,block>>>(d_a, d_b, d_c, count, offset);
	cudaDeviceSynchronize();

	cudaMemcpy(d_result, d_c, count*sizeof(float), cudaMemcpyDeviceToHost);

	bool indexFailed = false;
	for (int x = 0; x < count; x++) {
		if (h_result[x] != d_result[x]) {
			printf("error on index %d: %f, %f\n", x, h_result[x], d_result[x]);
			indexFailed = true;
		}
	}

	if (!indexFailed) {
		printf("all indices passed successfully\n");
	}

	return 0;
}