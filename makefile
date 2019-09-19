CC=g++
NVCC=/usr/local/cuda-10.1/bin/nvcc
NVPROF=/usr/local/cuda-10.1/bin/nvprof
MEMCHECK=/usr/local/cuda-10.1/bin/cuda-memcheck
CXXFLAGS=
CUDAFLAGS=-m64 -gencode arch=compute_30,code=compute_30
LIBS=-lcublas
LIBDIRS=/usr/lib/x86_64-linux-gnu:/usr/local/cuda-10.1/lib64
INCDIRS=

all: clean compile

clean:
	rm -rf bin

run:
	./bin/matrix_add.out
	./bin/matrix_inverse.out

compile:
	mkdir -p bin
	$(NVCC) $(CUDAFLAGS) ./src/matrix_add.cu -o ./bin/matrix_add.out
	$(NVCC) $(CUDAFLAGS) -L$(LIBDIRS) $(LIBS) ./src/matrix_inverse.cu -o ./bin/matrix_inverse.out

profile:
	sudo $(NVPROF) ./bin/matrix_add.out
	sudo $(NVPROF) ./bin/matrix_inverse.out

memcheck:
	$(MEMCHECK) ./bin/matrix_add.out
	$(MEMCHECK) ./bin/matrix_inverse.out