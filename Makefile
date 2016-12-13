SHELL:=/bin/bash
NVCC=nvcc

DFLAGS      = -lineinfo -w
INCFLAGS    = -I.
CFLAGS      = $(DFLAGS) $(INCFLAGS) -std=c++11
VERFLAGS	= -arch=compute_50 -code=sm_50
NVCCFLAGS   = $(CFLAGS) $(VERFLAGS) --ptxas-options=-v

DEPS		= kmeans_main.cu kmeans_utils.cpp kmeans_lloyd.cu file_io.cu

kmeans:
	$(NVCC) $(NVCCFLAGS) $(DEPS) -o kmeans_gpu
	
clean:
	rm -rf *.o kmeans_gpu
