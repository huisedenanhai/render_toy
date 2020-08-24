#pragma once

#include "math_consts.h"
#include "defs.h"
#include <cmath>
#include <type_traits>

// multiply
HOST_DEVICE_INLINE float3 operator*(const float3 &a, const float3 &b) {
  return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

HOST_DEVICE_INLINE float3 operator*(const float3 &a, float b) {
  return make_float3(a.x * b, a.y * b, a.z * b);
}

HOST_DEVICE_INLINE float3 operator*(float a, const float3 &b) {
  return make_float3(a * b.x, a * b.y, a * b.z);
}

HOST_DEVICE_INLINE void operator*=(float3 &a, const float3 &b) {
  a.x *= b.x;
  a.y *= b.y;
  a.z *= b.z;
}

HOST_DEVICE_INLINE void operator*=(float3 &a, float b) {
  a.x *= b;
  a.y *= b;
  a.z *= b;
}

HOST_DEVICE_INLINE float4 operator*(const float4 &a, const float4 &b) {
  return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

HOST_DEVICE_INLINE float4 operator*(const float4 &a, float b) {
  return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}

HOST_DEVICE_INLINE float4 operator*(float a, const float4 &b) {
  return make_float4(a * b.x, a * b.y, a * b.z, a * b.w);
}

HOST_DEVICE_INLINE void operator*=(float4 &a, const float4 &b) {
  a.x *= b.x;
  a.y *= b.y;
  a.z *= b.z;
  a.w *= b.w;
}

HOST_DEVICE_INLINE void operator*=(float4 &a, float b) {
  a.x *= b;
  a.y *= b;
  a.z *= b;
  a.w *= b;
}

// div
HOST_DEVICE_INLINE float3 operator/(const float3 &a, const float3 &b) {
  return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

HOST_DEVICE_INLINE float3 operator/(const float3 &a, float b) {
  return make_float3(a.x / b, a.y / b, a.z / b);
}

HOST_DEVICE_INLINE void operator/=(float3 &a, const float3 &b) {
  a.x /= b.x;
  a.y /= b.y;
  a.z /= b.z;
}

HOST_DEVICE_INLINE void operator/=(float3 &a, float b) {
  a.x /= b;
  a.y /= b;
  a.z /= b;
}

HOST_DEVICE_INLINE float4 operator/(const float4 &a, const float4 &b) {
  return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}

HOST_DEVICE_INLINE float4 operator/(const float4 &a, float b) {
  return make_float4(a.x / b, a.y / b, a.z / b, a.w / b);
}

HOST_DEVICE_INLINE void operator/=(float4 &a, const float4 &b) {
  a.x /= b.x;
  a.y /= b.y;
  a.z /= b.z;
  a.w /= b.w;
}

HOST_DEVICE_INLINE void operator/=(float4 &a, float b) {
  a.x /= b;
  a.y /= b;
  a.z /= b;
  a.w /= b;
}

// add
HOST_DEVICE_INLINE float3 operator+(const float3 &a, const float3 &b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

HOST_DEVICE_INLINE float3 operator+(const float3 &a, float b) {
  return make_float3(a.x + b, a.y + b, a.z + b);
}

HOST_DEVICE_INLINE float3 operator+(float a, const float3 &b) {
  return make_float3(a + b.x, a + b.y, a + b.z);
}

HOST_DEVICE_INLINE void operator+=(float3 &a, const float3 &b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
}

HOST_DEVICE_INLINE void operator+=(float3 &a, float b) {
  a.x += b;
  a.y += b;
  a.z += b;
}

HOST_DEVICE_INLINE float4 operator+(const float4 &a, const float4 &b) {
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

HOST_DEVICE_INLINE float4 operator+(const float4 &a, float b) {
  return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}

HOST_DEVICE_INLINE float4 operator+(float a, const float4 &b) {
  return make_float4(a + b.x, a + b.y, a + b.z, a + b.w);
}

HOST_DEVICE_INLINE void operator+=(float4 &a, const float4 &b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
}

HOST_DEVICE_INLINE void operator+=(float4 &a, float b) {
  a.x += b;
  a.y += b;
  a.z += b;
  a.w += b;
}

// sub
HOST_DEVICE_INLINE float3 operator-(const float3 &a) {
  return make_float3(-a.x, -a.y, -a.z);
}

HOST_DEVICE_INLINE float3 operator-(const float3 &a, const float3 &b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

HOST_DEVICE_INLINE float3 operator-(const float3 &a, float b) {
  return make_float3(a.x - b, a.y - b, a.z - b);
}

HOST_DEVICE_INLINE void operator-=(float3 &a, const float3 &b) {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
}

HOST_DEVICE_INLINE void operator-=(float3 &a, float b) {
  a.x -= b;
  a.y -= b;
  a.z -= b;
}

// others
HOST_DEVICE_INLINE float dot(const float3 &a, const float3 &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

HOST_DEVICE_INLINE float dot(const float4 &a, const float4 &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

HOST_DEVICE_INLINE float length(const float3 &v) {
  return sqrtf(dot(v, v));
}

HOST_DEVICE_INLINE float length(const float4 &v) {
  return sqrtf(dot(v, v));
}

HOST_DEVICE_INLINE float3 normalize(const float3 &v) {
  auto invLen = 1.0f / sqrtf(dot(v, v));
  return v * invLen;
}

HOST_DEVICE_INLINE float3 cross(const float3 &a, const float3 &b) {
  return make_float3(
      a.y * b.z - a.z * b.y, -a.x * b.z + a.z * b.x, a.x * b.y - a.y * b.x);
}

HOST_DEVICE_INLINE float3 faceforward(const float3 &n,
                                               const float3 &i,
                                               const float3 &nref) {
  return n * copysignf(1.0f, dot(i, nref));
}