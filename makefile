CC=g++
NVCC=/usr/local/cuda-10.1/bin/nvcc
NVPROF=/usr/local/cuda-10.1/bin/nvprof
MEMCHECK=/usr/local/cuda-10.1/bin/cuda-memcheck
CXXFLAGS=
CUDAFLAGS=--gpu-architecture=sm_50
LIBS=
LIBDIRS=
INCDIRS=
CURRENTFILE=array_sum

all: clean compile run

clean:
	rm -rf bin
	rm -rf dump

compile:
	mkdir -p bin
	$(NVCC) $(CUDAFLAGS) ./src/$(CURRENTFILE).cu -o ./bin/$(CURRENTFILE).out
run:
	./bin/$(CURRENTFILE).out

memory-check:
	$(MEMCHECK) ./bin/$(CURRENTFILE).out

profile:
	mkdir -p dump
	cd dump; sudo $(NVPROF) ../bin/$(CURRENTFILE).out 2>profile.log; cat profile.log;

profile-metrics:
	mkdir -p dump
	cd dump; sudo $(NVPROF) --metrics all ../bin/$(CURRENTFILE).out 2>profile-metrics.log; cat profile-metrics.log;

profile-events:
	mkdir -p dump
	cd dump; sudo $(NVPROF) --events all ../bin/$(CURRENTFILE).out 2>profile-events.log; cat profile-events.log;