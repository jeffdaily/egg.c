// Copyright (c) 2026 Advanced Micro Devices, Inc.
// Author: Jeff Daily <jeff.daily@amd.com>
#ifndef EGG_WARP_COMPAT_CUH
#define EGG_WARP_COMPAT_CUH

// EGGROLL maps one perturbation to one LOGICAL warp and partitions HIDDEN_DIM
// across exactly 32 lanes (loops stride by EGG_WARP_SIZE, per-lane arrays are
// sized MAX_STRIDE = HIDDEN_DIM_max / 32). EGG_WARP_SIZE is therefore a fixed
// 32-lane DATA-LAYOUT stride, not the physical wavefront width. It MUST stay 32
// on every arch (wave32 and wave64); using the runtime warpSize (64 on CDNA)
// would corrupt the data tiling. On wave64 two logical warps share one physical
// wavefront, so every shuffle/reduce below operates at explicit width 32 to keep
// the two logical warps independent. This is the "logical-warp" exception to the
// physical-warp host-query rule.
#define EGG_WARP_SIZE 32

#if defined(__HIP__)
// hipcc -x hip defines __HIP__ (not __HIP_PLATFORM_AMD__). On wave64 the lane
// mask must cover the physical wavefront width, so use a 64-bit all-ones mask;
// the explicit width=32 argument confines the op to the 32-lane logical warp.
#define EGG_FULL_MASK (~0ull)
#else
#define EGG_FULL_MASK 0xFFFFFFFFu
#endif

// hipCUB's WarpReduce<T> defaults LOGICAL_WARP_THREADS to the physical warp size
// (64 on gfx90a); pin it to the 32-lane logical warp so it sums the correct
// per-perturbation scope. Matches cub::WarpReduce<T,32> on NVIDIA.
template <typename T>
using EggWarpReduce = cub::WarpReduce<T, EGG_WARP_SIZE>;

__device__ __forceinline__ int eggShflSync(int v, int src_lane) {
    return __shfl_sync(EGG_FULL_MASK, v, src_lane, EGG_WARP_SIZE);
}

// Width-32 shuffle-down for the transformer attention head reductions: each head
// is reduced inside one 32-lane logical warp (off starts at 16). Pinning width to
// EGG_WARP_SIZE keeps the two logical warps that share a wave64 wavefront from
// mixing their partial sums; on NVIDIA / wave32 the explicit width is a no-op.
template <typename T>
__device__ __forceinline__ T eggShflDownSync(T v, int off) {
    return __shfl_down_sync(EGG_FULL_MASK, v, off, EGG_WARP_SIZE);
}

// Width-32 XOR shuffle for the RoPE neighbor-pair exchange (lane_mask 1). The
// partner is the adjacent lane within the same 32-lane logical warp; the explicit
// width plus the wavefront-wide mask keeps the swap inside the logical warp.
template <typename T>
__device__ __forceinline__ T eggShflXorSync(T v, int lane_mask) {
    return __shfl_xor_sync(EGG_FULL_MASK, v, lane_mask, EGG_WARP_SIZE);
}

__device__ __forceinline__ long long eggWarpBroadcast(long long val, int src_lane) {
    int lo = eggShflSync((int)val, src_lane);
    int hi = eggShflSync((int)(val >> 32), src_lane);
    return ((long long)hi << 32) | (unsigned int)lo;
}

#endif // EGG_WARP_COMPAT_CUH
