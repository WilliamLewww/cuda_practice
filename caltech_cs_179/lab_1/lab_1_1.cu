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

int main(void) {
	test1();
	test2();
	test3();
	test4();
	test5();

	return 0;
}