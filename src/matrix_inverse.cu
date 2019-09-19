#include <cublas_v2.h>

int main(void) {
	cublasStatus_t status;
    cublasHandle_t handle;

    status = cublasCreate(&handle);
    cublasDestroy(handle);

	return 0;
}