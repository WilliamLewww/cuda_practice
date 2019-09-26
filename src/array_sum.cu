#include <stdio.h>

__global__
void reduceNeighboredLess(int* g_i, int* g_o, int count) {
	
}

int main(void) {
	int count = 1 << 16;
	int size = count * sizeof(int);

	int* h_i = (int*)malloc(size);
	int* h_o = (int*)malloc(size);

	for (int x = 0; x < count; x++) { h_i[x] = x; }

	int *d_i, *d_o;
	cudaMalloc((float**)&d_i, size);
	cudaMalloc((float**)&d_o, size);

	cudaMemcpy(d_i, h_i, size, cudaMemcpyHostToDevice);

	dim3 block = 128;
	dim3 grid = ((count + block.x - 1) / block.x);
	reduceNeighboredLess<<<grid, block>>>(d_i, d_o, count);
	cudaDeviceSynchronize();

	cudaMemcpy(h_o, d_o, size, cudaMemcpyDeviceToHost);
	cudaFree(d_i);
	cudaFree(d_o);

	int sum = 0;
	for (int x = 0; x < grid.x + 5; x++) {
		sum += h_o[x];
	}

	printf("\ntotal sum (0 to %d): %d\n\n", count, sum);

	free(h_i);
	free(h_o);
	cudaDeviceReset();

	return 0;
}