// Copyright (c) 2026 Advanced Micro Devices, Inc.
// Author: Jeff Daily <jeff.daily@amd.com>
#ifndef EGG_HIP_COMPAT_CUH
#define EGG_HIP_COMPAT_CUH

// CUDA-spelling -> HIP compat shim for the raw-nvcc egg.c trainers. hipcc -x hip
// defines __HIP__ (not __HIP_PLATFORM_AMD__). This header lets the .cu sources
// keep their CUDA spelling: under HIP it pulls in the HIP runtime + hipCUB and
// maps the small, fixed set of CUDA runtime symbols / the cub namespace these
// files use. Under nvcc it expands to nothing, so the NVIDIA build is unchanged.
//
// rocThrust already exposes the thrust:: namespace at the same header paths, so
// thrust includes need no remap.

#if defined(__HIP__)

#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>

namespace cub = hipcub;

typedef hipError_t cudaError_t;
typedef hipError_t cudaError;
typedef hipDeviceProp_t cudaDeviceProp;

#define cudaSuccess hipSuccess
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMemset hipMemset
#define cudaMemcpy hipMemcpy
#define cudaMemcpyToSymbol hipMemcpyToSymbol
#define cudaMemcpyFromSymbol hipMemcpyFromSymbol
#define cudaGetSymbolAddress hipGetSymbolAddress
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice

// Multi-GPU and stream/async surface the replicated data-parallel mgpu trainer
// uses: each device gets its own stream, model copy and cuBLAS handle, fitness is
// aggregated through host pinned memory, then broadcast back per device. No NCCL,
// no peer access -- these are all 1:1 hip* runtime symbols.
typedef hipStream_t cudaStream_t;
#define cudaSetDevice hipSetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaStreamCreate hipStreamCreate
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaMallocHost hipHostMalloc
#define cudaFreeHost hipHostFree
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemsetAsync hipMemsetAsync

// hipBLAS (rocBLAS underneath) is column-major like cuBLAS, so the OP_T/OP_N and
// leading-dimension logic in the Muon Newton-Schulz iteration transfers 1:1. Only
// the small surface the trainer/Muon path touches is aliased; bodies stay guarded
// so the nvcc build keeps using cuBLAS unchanged.
#include <hipblas/hipblas.h>

typedef hipblasHandle_t cublasHandle_t;
typedef hipblasStatus_t cublasStatus_t;
typedef hipblasOperation_t cublasOperation_t;

#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define CUBLAS_OP_T HIPBLAS_OP_T
#define CUBLAS_OP_N HIPBLAS_OP_N
#define CUBLAS_POINTER_MODE_HOST HIPBLAS_POINTER_MODE_HOST
#define CUBLAS_POINTER_MODE_DEVICE HIPBLAS_POINTER_MODE_DEVICE

#define cublasCreate hipblasCreate
#define cublasDestroy hipblasDestroy
#define cublasSetStream hipblasSetStream
#define cublasSetPointerMode hipblasSetPointerMode
#define cublasSnrm2 hipblasSnrm2
#define cublasSgemm hipblasSgemm

// The transformer trainers use the CUDA 8-bit packed dot-product intrinsic
// __dp4a(a, b, c) = c + sum_i (int8)(a>>8i) * (int8)(b>>8i). ROCm exposes the
// amdgcn sdot4 builtin only under the dot1-insts target feature, which is not
// available on every supported arch (e.g. gfx1100), so provide a portable
// definition that the compiler lowers to the dot instruction where the hardware
// has it and to plain multiplies elsewhere. Same numeric result on every arch.
__device__ __forceinline__ int __dp4a(int a, int b, int c) {
    int res = c;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        int8_t av = (int8_t)((a >> (i * 8)) & 0xff);
        int8_t bv = (int8_t)((b >> (i * 8)) & 0xff);
        res += (int)av * (int)bv;
    }
    return res;
}

#endif // __HIP__

#endif // EGG_HIP_COMPAT_CUH
