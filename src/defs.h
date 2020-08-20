#pragma once

#ifdef __CUDACC__
#define HOST_DEVICE_DATA __device__
#else
#define HOST_DEVICE_DATA constexpr
#endif