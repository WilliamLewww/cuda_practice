#include <stdio.h>
#include <cstdlib>
#include <time.h>
#include <cmath>

void printMatrix(float* matrix, int rowCount, int columnCount) {
}

void initializeMatrix(float* matrix, int rowCount, int columnCount) {
	for (int y = 0; y < rowCount; y++) {
		for (int x = 0; x < columnCount; x++) {
			matrix[(y * columnCount) + x] = rand() % 25;
		}
	}
}

__global__
void addMatrix(float* out, float* a, float* b) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	out[x] = a[x] + b[x];
}

int main(void) {
	srand(time(NULL));

	int matrixRowCount = 4;
	int matrixColumnCount = 4;
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

	dim3 block(256);
	dim3 grid((matrixSize + block.x - 1) / block.x);
	addMatrix<<<grid, block>>>(d_c, d_a, d_b);

	cudaDeviceSynchronize();
	cudaMemcpy(h_c, d_c, memorySize, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	printMatrix(h_a, matrixSize);
	printMatrix(h_b, matrixSize);
	printMatrix(h_c, matrixSize);

	cudaDeviceReset();
	return 0;
}