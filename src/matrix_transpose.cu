#include <stdio.h>

__global__
void transposeReadRow(float* out, float* in, int rowCount, int columnCount) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	int idy = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (idx < columnCount && idy < rowCount) {
		out[idx*columnCount+idy] = in[idy*rowCount+idx];
	}
}

__global__
void transposeReadColumn(float* out, float* in, int rowCount, int columnCount) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	int idy = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (idx < columnCount && idy < rowCount) {
		out[idy*rowCount+idx] = in[idx*columnCount+idy];
	}
}

__global__
void transposeReadRowUnwrap8(float* out, float* in, int rowCount, int columnCount) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	int idy = (blockIdx.y * blockDim.y) + threadIdx.y;

	int x = idy*rowCount+idx;
	int y = idx*columnCount+idx;

	if (idx + 7 * blockDim.x < columnCount && idy < rowCount) {
		out[y] = in[x];
		out[y+rowCount*blockDim.x] = in[x+blockDim.x];
		out[y+2*rowCount*blockDim.x] = in[x+2*blockDim.x];
		out[y+3*rowCount*blockDim.x] = in[x+3*blockDim.x];
		out[y+4*rowCount*blockDim.x] = in[x+4*blockDim.x];
		out[y+5*rowCount*blockDim.x] = in[x+5*blockDim.x];
		out[y+6*rowCount*blockDim.x] = in[x+6*blockDim.x];
		out[y+7*rowCount*blockDim.x] = in[x+7*blockDim.x];

	}
}

__global__
void transposeReadColumnUnwrap8(float* out, float* in, int rowCount, int columnCount) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	int idy = (blockIdx.y * blockDim.y) + threadIdx.y;

	int x = idy*rowCount+idx;
	int y = idx*columnCount+idx;

	if (idx + 7 * blockDim.x < columnCount && idy < rowCount) {
		out[x] = in[y];
		out[x+blockDim.x] = in[y+rowCount*blockDim.x];
		out[x+2*blockDim.x] = in[y+2*rowCount*blockDim.x];
		out[x+3*blockDim.x] = in[y+3*rowCount*blockDim.x];
		out[x+4*blockDim.x] = in[y+4*rowCount*blockDim.x];
		out[x+5*blockDim.x] = in[y+5*rowCount*blockDim.x];
		out[x+6*blockDim.x] = in[y+6*rowCount*blockDim.x];
		out[x+7*blockDim.x] = in[y+7*rowCount*blockDim.x];
	}
}

__global__
void transposeDiagonalReadRow(float* out, float* in, int rowCount, int columnCount) {
	int blockX = (blockIdx.x + blockIdx.y) % gridDim.x;
	int blockY = blockIdx.x;

	int idx = (blockX * blockDim.x) + threadIdx.x;
	int idy = (blockY * blockDim.y) + threadIdx.y;

	if (idx < columnCount && idy < columnCount) {
		out[idx*columnCount+idy] = in[idy*rowCount+idx];
	}
}

__global__
void transposeDiagonalReadColumn(float* out, float* in, int rowCount, int columnCount) {
	int blockX = (blockIdx.x + blockIdx.y) % gridDim.x;
	int blockY = blockIdx.x;

	int idx = (blockX * blockDim.x) + threadIdx.x;
	int idy = (blockY * blockDim.y) + threadIdx.y;

	if (idx < columnCount && idy < columnCount) {
		out[idy*rowCount+idx] = in[idx*columnCount+idy];
	}
}

void transpose(float* out, float* in, int rowCount, int columnCount) {
	for (int y = 0; y < columnCount; y++) {
		for (int x = 0; x < rowCount; x++) {
			out[x*columnCount+y] = in[y*rowCount+x];
		}
	}
}

bool checkMatrix(float* matrixA, float* matrixB, int rowCount, int columnCount) {
	for (int x = 0; x < rowCount * columnCount; x++) {
		if (matrixA[x] != matrixB[x]) {
			return false;
		}
	}

	return true;
}

void printMatrix(float* matrix, int rowCount, int columnCount) {
	for (int y = 0; y < columnCount; y++) {
		for (int x = 0; x < rowCount; x++) {
			printf("%-3d ", int(matrix[y*rowCount+x]));
		}
		printf("\n");
	}
}

int main(void) {
	printf("\n");
	int rowCount = 64;
	int columnCount = 64;

	dim3 block(32,32);
	dim3 grid((columnCount+block.x-1)/block.x, (rowCount+block.y-1)/block.y);
	dim3 gridUnwrap8((columnCount+(block.x/8)-1)/(block.x/8), (rowCount+block.y-1)/block.y);

	float* h_matrix = (float*)malloc(rowCount*columnCount*sizeof(float));
	for (int x = 0; x < rowCount * columnCount; x++) { h_matrix[x] = x; }
	float* h_transpose_matrix = (float*)malloc(rowCount*columnCount*sizeof(float));

	float* expected_matrix = (float*)malloc(rowCount*columnCount*sizeof(float));
	transpose(expected_matrix, h_matrix, rowCount, columnCount);

	float* d_matrix;
	cudaMalloc((float**)&d_matrix, rowCount*columnCount*sizeof(float));
	cudaMemcpy(d_matrix, h_matrix, rowCount*columnCount*sizeof(float), cudaMemcpyHostToDevice);

	float* d_transpose_matrix;
	cudaMalloc((float**)&d_transpose_matrix, rowCount*columnCount*sizeof(float));

	transposeReadRow<<<grid,block>>>(d_transpose_matrix, d_matrix, rowCount, columnCount);
	cudaDeviceSynchronize();
	cudaMemcpy(h_transpose_matrix, d_transpose_matrix, rowCount*columnCount*sizeof(float), cudaMemcpyDeviceToHost);
	printf("%-30s: %d\n", "transposeReadRow", checkMatrix(expected_matrix, h_transpose_matrix, rowCount, columnCount));

	transposeReadColumn<<<grid,block>>>(d_transpose_matrix, d_matrix, rowCount, columnCount);
	cudaDeviceSynchronize();
	cudaMemcpy(h_transpose_matrix, d_transpose_matrix, rowCount*columnCount*sizeof(float), cudaMemcpyDeviceToHost);
	printf("%-30s: %d\n", "transposeReadColumn", checkMatrix(expected_matrix, h_transpose_matrix, rowCount, columnCount));

	transposeReadRowUnwrap8<<<gridUnwrap8,block>>>(d_transpose_matrix, d_matrix, rowCount, columnCount);
	cudaDeviceSynchronize();
	cudaMemcpy(h_transpose_matrix, d_transpose_matrix, rowCount*columnCount*sizeof(float), cudaMemcpyDeviceToHost);
	printf("%-30s: %d\n", "transposeReadRowUnwrap8", checkMatrix(expected_matrix, h_transpose_matrix, rowCount, columnCount));

	transposeReadColumnUnwrap8<<<gridUnwrap8,block>>>(d_transpose_matrix, d_matrix, rowCount, columnCount);
	cudaDeviceSynchronize();
	cudaMemcpy(h_transpose_matrix, d_transpose_matrix, rowCount*columnCount*sizeof(float), cudaMemcpyDeviceToHost);
	printf("%-30s: %d\n", "transposeReadColumnUnwrap8", checkMatrix(expected_matrix, h_transpose_matrix, rowCount, columnCount));

	transposeDiagonalReadRow<<<gridUnwrap8,block>>>(d_transpose_matrix, d_matrix, rowCount, columnCount);
	cudaDeviceSynchronize();
	cudaMemcpy(h_transpose_matrix, d_transpose_matrix, rowCount*columnCount*sizeof(float), cudaMemcpyDeviceToHost);
	printf("%-30s: %d\n", "transposeDiagonalReadRow", checkMatrix(expected_matrix, h_transpose_matrix, rowCount, columnCount));

	transposeDiagonalReadColumn<<<gridUnwrap8,block>>>(d_transpose_matrix, d_matrix, rowCount, columnCount);
	cudaDeviceSynchronize();
	cudaMemcpy(h_transpose_matrix, d_transpose_matrix, rowCount*columnCount*sizeof(float), cudaMemcpyDeviceToHost);
	printf("%-30s: %d\n", "transposeDiagonalReadColumn", checkMatrix(expected_matrix, h_transpose_matrix, rowCount, columnCount));

	cudaFree(d_matrix);
	cudaFree(d_transpose_matrix);
	free(h_matrix);
	free(h_transpose_matrix);
	free(expected_matrix);
	cudaDeviceReset();

	printf("\n");

	return 0;
}