CC=g++
NVCC=nvcc
CXXFLAGS=
CUDAFLAGS=
LIBS=
LIBDIRS=
INCDIRS=

matrix_add: matrix_add.cu
	$(NVCC) $(CUDAFLAGS) matrix_add.cu -o matrix_add.out
clean:
	rm -rf test *.out