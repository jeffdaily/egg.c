NVCC = nvcc
CXX = g++
CFLAGS = -O3 -Iinclude

# GPU toolchain. The default is the upstream nvcc build; pass USE_HIP=1 to build
# the worker with hipcc for AMD ROCm. HIPARCH selects the GPU target(s) and is
# caller-overridable (e.g. HIPARCH="gfx90a gfx1100" for a fat binary); it is not
# hardcoded. The single worker translation unit (worker.cu #includes kernels.cu)
# needs no relocatable device code, so -fgpu-rdc is omitted on the HIP arm.
ifeq ($(USE_HIP),1)
  HIPARCH ?= gfx90a
  GPUCC = hipcc
  GPUFLAGS = -O3 -Iinclude -x hip $(addprefix --offload-arch=,$(HIPARCH)) -lhipblas
else
  GPUCC = $(NVCC)
  GPUFLAGS = -O3 -Iinclude -arch=sm_86 -lcublas -rdc=true
endif

all: coordinator worker print_arch

print_arch: print_arch.cpp
	$(CXX) $(CFLAGS) -o print_arch print_arch.cpp

coordinator: src/coordinator.cpp
	$(CXX) $(CFLAGS) -o coordinator src/coordinator.cpp -lpthread

worker: src/worker.cu
	$(GPUCC) $(GPUFLAGS) -o worker src/worker.cu

clean:
	rm -f coordinator worker print_arch
