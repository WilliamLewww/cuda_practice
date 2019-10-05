#include <stdio.h>

void test1() {
    int* a = new int;
    *a = 3;
    *a = *a + 2;
    printf("%d\n", *a);
}

void test2() {
    int* a = (int*)malloc(sizeof(int));
    int* b = (int*)malloc(sizeof(int));

    if (!(a && b)) {
        printf("Out of memory\n");
        exit(-1);
    }
    *a = 2;
    *b = 3;
}

void test3() {
    int i = 5;
    int* a = (int*)malloc(1000*sizeof(int));

    if (!a) {
        printf("Out of memory\n");
        exit(-1);
    }
    *(i + a) = i; 
}

void test4() {
    int** a = (int**)malloc(3*sizeof(int*));
    a[0] = (int*)malloc(100*sizeof(int*));
    a[1] = (int*)malloc(100*sizeof(int*));
    a[2] = (int*)malloc(100*sizeof(int*));

    a[1][1] = 5;
}

void test5() {
    int* a = (int*)malloc(sizeof(int));
    scanf("%d", a);
    if (!*a) {
        printf("Value is 0\n");
    }
}

void question1() {
    /*
        y_1[n] = x[n - 1] + x[n] + x[n + 1]
        y_2[n] = y_2[n - 2] + y_2[n - 1] + x[n]

        The second implementation would be harder because we would need to syncronize the GPU 
        code due to the sequential nature of the statement. The first implementation is better 
        because it does not require syncronization of threads and can all be done in parallel.

    */
}

void question2() {
    /*
        y[n] = c * x[n] + (1 - c) * y[n - 1]

        The code is not capable of running in parallel because it requires the previous iteration
        of the function. If c is close to 1, we could ignore the "y[n - 1]" part of the code 
        because it will be close to 0. In this case if we drop the last part of the calculation
        we could easily implement the code in a parallel manner.
    */
}

int main(void) {
	test1();
	test2();
	test3();
	test4();
	test5();

	return 0;
}