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
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost

#endif // __HIP__

#endif // EGG_HIP_COMPAT_CUH
