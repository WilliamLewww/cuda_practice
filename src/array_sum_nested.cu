//input: integer
//output: sum of all numbers in [0, integer)

#include <stdio.h>
#include "analysis.h"

__global__
void nestedHelloWorld(int count, int depth) {
	printf("Depth %d, Thread %d: Hello World!\n", depth, threadIdx.x);

	if (count == 1) { return; }

	int threadCount = count >> 1;
	if (threadIdx.x == 0 && threadCount > 0) {
		printf("\n");
		nestedHelloWorld<<<1, threadCount>>>(threadCount, ++depth);
	}
}

int main(void) {
	nestedHelloWorld<<<1,8>>>(8, 0);
	cudaDeviceSynchronize();
	cudaDeviceReset();

	return 0;
}