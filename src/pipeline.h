#pragma once

#include <optix.h>

namespace toy {
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
  float3 color;
};

struct HitGroupData {
  float3 baseColor;
  float3 emission;
};
} // namespace toy