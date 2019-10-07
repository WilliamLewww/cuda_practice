#include <stdio.h>

void question1() {
	/*
		latency of arithmetic instruction = ~10+ ns
		GK110 SM (Kepler) has:
			2 warp schedulers
			4 warp dispatchers
		
		2 * 4 * 10 = 80 arithmetic instructions required
	*/
}

void question2() {
	/*
		block shape = (32, 32, 1)

		a)
			int idx = threadIdx.y + blockSize.y * threadIdx.x;
			if (idx % 32 < 16) {
			    foo();
			}
			else {
			    bar();
			}

		The code will not diverge because the idx will always be in iterations of 32; The block
		size is 32x32x1 meaning every iteration of threadIdx.x will increment idx by a scale of
		32. threadIdx.y will stay consistant for every thread inside the warp meaning the warp
		will not diverge
		
		b)
			const float pi = 3.14;
			float result = 1.0;
			for (int i = 0; i < threadIdx.x; i++) {
			    result *= pi;
			}

		The code will diverge because each thread in the warp has a different threadIdx.x meaning
		threads with smaller thread indices will have to stall and wait for the threads with 
		larger thread indices
	*/
}

void question3() {
	/*
		block shape = (32, 32, 1)

		a)
			data[threadIdx.x + blockSize.x * threadIdx.y] = 1.0;

		b)
			data[threadIdx.y + blockSize.y * threadIdx.x] = 1.0;

		c)
			data[1 + threadIdx.x + blockSize.x * threadIdx.y] = 1.0;
	*/
}

int main(void) {

}