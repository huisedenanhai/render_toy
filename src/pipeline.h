#pragma once

#include <optix.h>

namespace dev {
// coordinate setting:
// right hand coordinate, y points upward
// for 2d space, origin point is at left bottom, y points up, x points right

// in a common 2d space, a Rect got the following sematics:
// (x, y)
// (x, y + height)-----------(x + width, y + height)
//    |                             |
// (x, y)--------------------(x + width, y)
struct Rect {
  float x, y;
  float width, height;
};

struct RectInt {
  int x, y;
  int width, height;
};

__host__ __device__ inline Rect
make_rect(float x, float y, float width, float height) {
  Rect r;
  r.x = x;
  r.y = y;
  r.width = width;
  r.height = height;
  return r;
}

__host__ __device__ inline RectInt
make_rect_int(int x, int y, int width, int height) {
  RectInt r;
  r.x = x;
  r.y = y;
  r.width = width;
  r.height = height;
  return r;
}

struct Camera {
  // camera position and axis are all in world space
  float3 position;
  // right hand coordinate forming camera local space
  // right, up, back corresponds to x, y, z
  // these vectors are assumed to be normalized
  float3 right;
  float3 up;
  float3 back;
  // canvas specified in camera local space,
  // the canvas is at +1 front, a.k.a -1 back
  Rect canvas;
};

struct Frame {
  float3 *buffer;
  unsigned int width, height;
  // tile is in frame coordinate
  // (x, y)
  // (0, height)-----------(width, height)
  //    |                       |
  // (0, 0)----------------(width, 0)
  // tile should not be outside of the output frame
  RectInt tile;
};

struct Scene {
  OptixTraversableHandle gas;
  float epsilon;
  // assume the scene can be enclosed by a sphere at world origin with diameter
  // 'extent'
  float extent;
};

struct LaunchParams {
  Camera camera;
  // the whole output frame fits with the uncropped canvas of camera
  Frame outputFrame;
  Scene scene;

  int spp;
  int maxPathLength;
};

struct RayGenData {};

struct ExceptionData {
  float3 errorColor;
};

struct MissData {
  float3 colorCoeff;
};

struct DiffuseHitGroupData {
  float3 baseColorCoeff;
  float3 emissionCoeff;
};

struct GlassHitGroupData {
  float3 baseColorCoeff;
  struct IOR {
    float waveLength390;
    float waveLength830;
  } ior;
};

struct BlackBodyHitGroupData {
  // temperature in K
  float temperature;
  float scaleFactor;

  constexpr static float c = 299792458.0f;
  constexpr static float h = 6.62606957e-34f;
  constexpr static float kb = 1.3806488e-23f;

  __host__ __device__ inline float unscaled_total_energy() {
    constexpr float factor = 1.80494e-8f;
    float t2 = temperature * temperature;
    float t4 = t2 * t2;
    return factor * t4;
  }

  // wave length is with the unit nm
  // sample without normalization
  __host__ __device__ inline float sample_spectrum_unscaled(float waveLength) {
    constexpr float twohc2 = 1.19104e-16f;
    constexpr float hc_k = 0.0143878f;
    // convert to meter
    float lm = waveLength * 1e-9f;
    float lm2 = lm * lm;
    float lm5 = lm2 * lm2 * lm;
    return twohc2 / (lm5 * (expf(hc_k / (lm * temperature)) - 1));
  }

  __host__ __device__ inline float sample_spectrum_scaled(float waveLength) {
    return sample_spectrum_unscaled(waveLength) * scaleFactor;
  }
};

__host__ __device__ inline float multiply_and_add(float a, float b, float c) {
  return a * b + c;
}

__host__ __device__ inline float eval_spectrum(const float3 &coeff,
                                               float waveLength) {
  float x = multiply_and_add(
      multiply_and_add(coeff.x, waveLength, coeff.y), waveLength, coeff.z);
  float y = 1.f / sqrtf(multiply_and_add(x, x, 1.f));
  return multiply_and_add(.5f * x, y, .5f);
}

} // namespace dev