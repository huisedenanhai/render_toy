#pragma once

#ifdef __CUDACC__
#define HOST_DEVICE_DATA __device__
#define HOST_DEVICE_INLINE __host__ __device__ __forceinline__
#else
#define HOST_DEVICE_DATA constexpr
#define HOST_DEVICE_INLINE inline
#endif