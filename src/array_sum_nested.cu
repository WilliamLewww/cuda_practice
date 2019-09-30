//input: integer
//output: sum of all numbers in [0, integer)

#include <stdio.h>
#include "analysis.h"

__global__
void nestedHelloWorld(int count, int depth) {
	printf("Depth %d, Thread %d, Block %d: Hello World!\n", depth, threadIdx.x, blockIdx.x);

	if (count == 1) { return; }

	int threadCount = count >> 1;
	if (threadIdx.x == 0 && threadCount > 0) {
		printf("\n");
		nestedHelloWorld<<<1, threadCount>>>(threadCount, ++depth);
	}
}

void callNestedHelloWord() {
	int block = 8;
	int grid = 1;

	nestedHelloWorld<<<grid,block>>>(block, 0);
	cudaDeviceSynchronize();
	cudaDeviceReset();
}

int main(void) {
	callNestedHelloWord();
	
	return 0;
}