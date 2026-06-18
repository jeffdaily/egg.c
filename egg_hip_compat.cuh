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
