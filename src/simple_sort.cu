#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void printArray(int* array, int count) {
	for (int x = 0; x < count; x++) {
		printf("%d ", array[x]);
		if (x % 15 == 0) { printf("\n"); }
	}
}

int main(void) {
	printf("\n");

	srand(time(NULL));

	int count = 5000;
	dim3 block(32);
	dim3 grid((count + block.x - 1) / block.x);

	int* host_array = (int*)malloc(count*sizeof(int));
	for (int x = 0; x < count; x++) { host_array[x] = rand() % count; }

	printf("\n");
	return 0;
}