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
	return 0;
}