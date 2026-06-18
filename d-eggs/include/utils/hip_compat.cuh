// Copyright (c) 2026 Advanced Micro Devices, Inc.
// Author: Jeff Daily <jeff.daily@amd.com>
#ifndef EGG_DEGGS_HIP_COMPAT_CUH
#define EGG_DEGGS_HIP_COMPAT_CUH

// CUDA-spelling -> HIP compat shim for the d-eggs distributed trainer. d-eggs
// has its own include tree (-Iinclude), so it carries a self-contained shim
// rather than reaching into the single-GPU trainers' headers. hipcc -x hip
// defines __HIP__; under it this header pulls in the HIP runtime, hipCUB and
// hipBLAS and maps the fixed set of CUDA runtime / cuBLAS symbols and the cub
// namespace the worker, kernels and model/optimizer headers use. Under nvcc it
// expands to nothing, so the NVIDIA build is unchanged.
//
// rocThrust exposes thrust:: at the same header paths, so thrust needs no remap.

#if defined(__HIP__)

#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#include <hipblas/hipblas.h>
#include "../config.h"

namespace cub = hipcub;

typedef hipError_t cudaError_t;
typedef hipError_t cudaError;
typedef hipDeviceProp_t cudaDeviceProp;
typedef hipPointerAttribute_t cudaPointerAttributes;

#define cudaSuccess hipSuccess
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaSetDevice hipSetDevice
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaPointerGetAttributes hipPointerGetAttributes

#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMallocHost hipHostMalloc
#define cudaFreeHost hipHostFree
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyToSymbol hipMemcpyToSymbol
#define cudaMemcpyFromSymbol hipMemcpyFromSymbol
#define cudaGetSymbolAddress hipGetSymbolAddress
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice

// Stream surface for the captured optimizer update.
typedef hipStream_t cudaStream_t;
#define cudaStreamCreate hipStreamCreate
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize

// Managed (unified) memory: the worker allocates the model, Adam state and
// dataset as managed buffers with a preferred location and prefetch so the
// host can read the host-coherent atomic update counter between graph launches.
// These all exist on ROCm with the same semantics.
#define cudaMallocManaged hipMallocManaged
#define cudaMemAdvise hipMemAdvise
#define cudaMemPrefetchAsync hipMemPrefetchAsync
#define cudaMemAdviseSetPreferredLocation hipMemAdviseSetPreferredLocation
#define cudaMemAdviseSetAccessedBy hipMemAdviseSetAccessedBy

// Graph capture/replay for the optimizer step. The captured region issues only
// async kernel launches and async memsets/copies on one stream (no host sync,
// no cuBLAS unless USE_MUON), so it is capture-safe; these map 1:1.
typedef hipGraph_t cudaGraph_t;
typedef hipGraphExec_t cudaGraphExec_t;
typedef hipGraphNode_t cudaGraphNode_t;
#define cudaStreamCaptureModeGlobal hipStreamCaptureModeGlobal
#define cudaStreamBeginCapture hipStreamBeginCapture
#define cudaStreamEndCapture hipStreamEndCapture
#define cudaGraphInstantiate hipGraphInstantiate
#define cudaGraphGetNodes hipGraphGetNodes
#define cudaGraphLaunch hipGraphLaunch
#define cudaGraphDestroy hipGraphDestroy
#define cudaGraphExecDestroy hipGraphExecDestroy

// hipBLAS (rocBLAS underneath) is column-major like cuBLAS, so the OP_T/OP_N
// and leading-dimension logic in the Muon Newton-Schulz iteration transfers
// 1:1. Only the surface the Muon path touches is aliased.
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

// EGGROLL maps one perturbation to one LOGICAL warp and partitions the hidden
// dimension across exactly 32 lanes (WARP_SIZE in config.h is a fixed 32-lane
// data-layout stride, not the physical wavefront). On wave64 (gfx90a) two
// logical warps share one wavefront, so the attention head reductions and the
// RoPE neighbor-pair exchange must run at explicit width 32 with a wavefront-
// wide mask to keep the two logical warps independent. The 64-bit all-ones mask
// covers the physical wavefront; the width=32 argument confines the op to the
// logical warp. On NVIDIA / wave32 the explicit width is a no-op.
#define EGG_FULL_MASK (~0ull)

template <typename T>
__device__ __forceinline__ T eggShflDownSync(T v, int off) {
    return __shfl_down_sync(EGG_FULL_MASK, v, off, WARP_SIZE);
}

template <typename T>
__device__ __forceinline__ T eggShflXorSync(T v, int lane_mask) {
    return __shfl_xor_sync(EGG_FULL_MASK, v, lane_mask, WARP_SIZE);
}

// The int8 packed dot-product intrinsic used by the model matmuls. The amdgcn
// sdot4 builtin needs the dot-insts target feature, absent on some supported
// arches (e.g. gfx1100), so provide a portable definition the compiler lowers
// to the dot instruction where present and to plain multiplies elsewhere. Same
// numeric result on every arch.
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

#endif // EGG_DEGGS_HIP_COMPAT_CUH
