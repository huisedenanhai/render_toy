#pragma once

#include <cmath>

#define VEC_MATH_HOST_DEVICE_INLINE __host__ __device__ __forceinline__

constexpr float Pi = 3.1415926f;
constexpr float InvPi = 1.0f / 3.1415926f;

// multiply
VEC_MATH_HOST_DEVICE_INLINE float3 operator*(const float3 &a, const float3 &b) {
  return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

VEC_MATH_HOST_DEVICE_INLINE float3 operator*(const float3 &a, float b) {
  return make_float3(a.x * b, a.y * b, a.z * b);
}

VEC_MATH_HOST_DEVICE_INLINE float3 operator*(float a, const float3 &b) {
  return make_float3(a * b.x, a * b.y, a * b.z);
}

VEC_MATH_HOST_DEVICE_INLINE void operator*=(float3 &a, const float3 &b) {
  a.x *= b.x;
  a.y *= b.y;
  a.z *= b.z;
}

VEC_MATH_HOST_DEVICE_INLINE void operator*=(float3 &a, float b) {
  a.x *= b;
  a.y *= b;
  a.z *= b;
}

// div
VEC_MATH_HOST_DEVICE_INLINE float3 operator/(const float3 &a, const float3 &b) {
  return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

VEC_MATH_HOST_DEVICE_INLINE float3 operator/(const float3 &a, float b) {
  return make_float3(a.x / b, a.y / b, a.z / b);
}

VEC_MATH_HOST_DEVICE_INLINE void operator/=(float3 &a, const float3 &b) {
  a.x /= b.x;
  a.y /= b.y;
  a.z /= b.z;
}

VEC_MATH_HOST_DEVICE_INLINE void operator/=(float3 &a, float b) {
  a.x /= b;
  a.y /= b;
  a.z /= b;
}

// add
VEC_MATH_HOST_DEVICE_INLINE float3 operator+(const float3 &a, const float3 &b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

VEC_MATH_HOST_DEVICE_INLINE float3 operator+(const float3 &a, float b) {
  return make_float3(a.x + b, a.y + b, a.z + b);
}

VEC_MATH_HOST_DEVICE_INLINE float3 operator+(float a, const float3 &b) {
  return make_float3(a + b.x, a + b.y, a + b.z);
}

VEC_MATH_HOST_DEVICE_INLINE void operator+=(float3 &a, const float3 &b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
}

VEC_MATH_HOST_DEVICE_INLINE void operator+=(float3 &a, float b) {
  a.x += b;
  a.y += b;
  a.z += b;
}

// sub
VEC_MATH_HOST_DEVICE_INLINE float3 operator-(const float3 &a) {
  return make_float3(-a.x, -a.y, -a.z);
}

VEC_MATH_HOST_DEVICE_INLINE float3 operator-(const float3 &a, const float3 &b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

VEC_MATH_HOST_DEVICE_INLINE float3 operator-(const float3 &a, float b) {
  return make_float3(a.x - b, a.y - b, a.z - b);
}

VEC_MATH_HOST_DEVICE_INLINE void operator-=(float3 &a, const float3 &b) {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
}

VEC_MATH_HOST_DEVICE_INLINE void operator-=(float3 &a, float b) {
  a.x -= b;
  a.y -= b;
  a.z -= b;
}

// others
VEC_MATH_HOST_DEVICE_INLINE float dot(const float3 &a, const float3 &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

VEC_MATH_HOST_DEVICE_INLINE float length(const float3 &v) {
  return sqrtf(dot(v, v));
}

VEC_MATH_HOST_DEVICE_INLINE float3 normalize(const float3 &v) {
  auto invLen = 1.0f / sqrtf(dot(v, v));
  return v * invLen;
}

VEC_MATH_HOST_DEVICE_INLINE float3 cross(const float3 &a, const float3 &b) {
  return make_float3(
      a.y * b.z - a.z * b.y, -a.x * b.z + a.z * b.x, a.x * b.y - a.y * b.x);
}

VEC_MATH_HOST_DEVICE_INLINE float3 faceforward(const float3 &n,
                                               const float3 &i,
                                               const float3 &nref) {
  return n * copysignf(1.0f, dot(i, nref));
}