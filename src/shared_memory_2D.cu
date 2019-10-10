#include <stdio.h>
#define MATRIX_ROWS 5
#define MATRIX_COLUMNS 5

__global__
void createMatrixStatic(float* out) {
	__shared__ float matrix[MATRIX_ROWS][MATRIX_COLUMNS];

	int idx = blockIdx.y * blockDim.x + threadIdx.x;
	int idy = blockIdx.x * blockDim.y + threadIdx.y;

	if (idx < MATRIX_COLUMNS && idy < MATRIX_ROWS) {
		matrix[idy][idx] = idx + idy;
		out[idy*MATRIX_COLUMNS+idx] = matrix[idy][idx];
	}
}

void printMatrix(float* matrix) {
	for (int y = 0; y < MATRIX_COLUMNS; y++) {
		for (int x = 0; x < MATRIX_ROWS; x++) {
			printf("%-3d ", int(matrix[y*MATRIX_ROWS+x]));
		}
		printf("\n");
	}
}

int main(void) {
	printf("\n");

	dim3 block(MATRIX_ROWS, MATRIX_COLUMNS);
	dim3 grid((MATRIX_ROWS+block.x-1)/block.x, (MATRIX_COLUMNS+block.y-1)/block.y);

	float* host_matrix = (float*)malloc(MATRIX_ROWS*MATRIX_COLUMNS*sizeof(float));

	float* device_matrix;
	cudaMalloc((float**)&device_matrix, MATRIX_ROWS*MATRIX_COLUMNS*sizeof(float));

	createMatrixStatic<<<grid,block>>>(device_matrix);
	cudaDeviceSynchronize();
	cudaMemcpy(host_matrix, device_matrix, MATRIX_ROWS*MATRIX_COLUMNS*sizeof(float), cudaMemcpyDeviceToHost);
	printMatrix(host_matrix);

	printf("\n");
	return 0;
}