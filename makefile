CUDAPATH=/usr/local/cuda-10.1

CC=g++
NVCC=$(CUDAPATH)/bin/nvcc
NVPROF=$(CUDAPATH)/bin/nvprof
MEMCHECK=$(CUDAPATH)/bin/cuda-memcheck
NSIGHTCLI=$(CUDAPATH)/bin/nv-nsight-cu-cli
CUDAFLAGS=--gpu-architecture=sm_50 -rdc=true
CURRENTFILE=array_sum_nested

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

nsight-cli:
	mkdir -p dump
	cd dump; sudo $(NSIGHTCLI) ../bin/$(CURRENTFILE).out > nsight-cli.log; cat nsight-cli.log;