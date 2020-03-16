#include <stdio.h>
#include <cstdlib>
#include <time.h>
#include <cmath>

void printMatrix(float* matrix, int rowCount, int columnCount) {
	for (int y = 0; y < rowCount; y++) {
		for (int x = 0; x < columnCount; x++) {
			printf("%d ", int(matrix[(y * columnCount) + x]));
		}
		printf("\n");
	}
	printf("\n");
}

void initializeMatrix(float* matrix, int rowCount, int columnCount) {
	for (int y = 0; y < rowCount; y++) {
		for (int x = 0; x < columnCount; x++) {
			matrix[(y * columnCount) + x] = rand() % 25;
		}
	}
}

__global__
void addMatrix(float* out, float* a, float* b, int rowCount, int columnCount) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	int index = (x * columnCount) + y;

	if (x < rowCount && y < columnCount) {
		out[index] = a[index] + b[index];
	}
}

int main(void) {
	srand(time(NULL));

	int matrixRowCount = 1<<5;
	int matrixColumnCount = 1<<5;
	int memorySize = matrixRowCount*matrixColumnCount*sizeof(float);

	float* h_a = (float*)malloc(memorySize);
	float* h_b = (float*)malloc(memorySize);
	float* h_c = (float*)malloc(memorySize);

	initializeMatrix(h_a, matrixRowCount, matrixColumnCount);
	initializeMatrix(h_b, matrixRowCount, matrixColumnCount);

	float *d_a, *d_b, *d_c;
	cudaMalloc((float**)&d_a, memorySize);
	cudaMalloc((float**)&d_b, memorySize);
	cudaMalloc((float**)&d_c, memorySize);

	cudaMemcpy(d_a, h_a, memorySize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, memorySize, cudaMemcpyHostToDevice);

	dim3 block(32, 32);
	dim3 grid((matrixRowCount + block.x - 1) / block.x, (matrixColumnCount + block.y - 1) / block.y);
	addMatrix<<<grid, block>>>(d_c, d_a, d_b, matrixRowCount, matrixColumnCount);

	cudaDeviceSynchronize();
	cudaMemcpy(h_c, d_c, memorySize, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	// printMatrix(h_a, matrixRowCount, matrixColumnCount);
	// printMatrix(h_b, matrixRowCount, matrixColumnCount);
	// printMatrix(h_c, matrixRowCount, matrixColumnCount);

	cudaDeviceReset();
	return 0;
}