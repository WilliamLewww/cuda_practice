#include <stdio.h>

struct AoS {
	float x;
	float y;
};

struct SoA {
	float* x;
	float* y;
};

void misalignedReadHost(float* arrayA, float* arrayB, float* arrayC, int count, int offset) {
	for (int offsetIdx = offset, idx = 0; offsetIdx < count; offsetIdx++, idx++) {
		arrayC[idx] = arrayA[offsetIdx] + arrayB[offsetIdx];
	}
}

void misalignedWriteHost(float* arrayA, float* arrayB, float* arrayC, int count, int offset) {
	for (int offsetIdx = offset, idx = 0; offsetIdx < count; offsetIdx++, idx++) {
		arrayC[offsetIdx] = arrayA[idx] + arrayB[idx];
	}
}

__global__
void arrayOfStructuresTest(AoS* arrayA, AoS* arrayB, int count) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (idx < count) {
		arrayB[idx].x = arrayA[idx].x * 2;
		arrayB[idx].y = arrayA[idx].y * 2;
	}
}

__global__
void structureOfArraysTest(SoA* structureA, SoA* structureB, int count) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (idx < count) {
		structureB->x[idx] = structureA->x[idx] * 2;
		structureB->y[idx] = structureA->y[idx] * 2; 
	}
}

__global__
void misalignedReadDevice(float* arrayA, float* arrayB, float* arrayC, int count, int offset) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	int offsetIdx = idx + offset;

	if (offsetIdx < count) {
		arrayC[idx] = arrayA[offsetIdx] + arrayB[offsetIdx];
	}
}

__global__
void misalignedWriteDevice(float* arrayA, float* arrayB, float* arrayC, int count, int offset) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	int offsetIdx = idx + offset;

	if (offsetIdx < count) {
		arrayC[offsetIdx] = arrayA[idx] + arrayB[idx];
	}
}

int main(void) {
	printf("\n");
	
	int count = 1 << 16;
	dim3 block = (512);
	dim3 grid = ((count + block.x - 1) / block.x);

	int offset = 11;

	float* h_a = (float*)malloc(count*sizeof(float));
	float* h_b = (float*)malloc(count*sizeof(float));

	for (int x = 0; x < count; x++) {
		h_a[x] = x * 2;
		h_b[x] = (x * 2) + 1;
	}

	float* h_result = (float*)malloc(count*sizeof(float));
	float* d_result = (float*)malloc(count*sizeof(float));

	misalignedReadHost(h_a, h_b, h_result, count, offset);

	float *d_a, *d_b, *d_c;
	cudaMalloc((float**)&d_a, count*sizeof(float));
	cudaMemcpy(d_a, h_a, count*sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc((float**)&d_b, count*sizeof(float));
	cudaMemcpy(d_b, h_b, count*sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc((float**)&d_c, count*sizeof(float));

	misalignedReadDevice<<<grid,block>>>(d_a, d_b, d_c, count, offset);
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