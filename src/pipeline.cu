#include "pipeline.h"
#include "vec_math.h"
#include <optix.h>

using namespace toy;

extern "C" {
__constant__ LaunchParams g_LaunchParams;
}

// Generate random unsigned int in [0, 2^24)
__device__ __forceinline__ unsigned int lcg(unsigned int &prev) {
  constexpr int LCG_A = 1664525u;
  constexpr int LCG_C = 1013904223u;
  prev = (LCG_A * prev + LCG_C);
  return prev & 0x00FFFFFF;
}

// rand float in range [0, 1)
__device__ __forceinline__ float rnd(unsigned int &randState) {
  return (float)lcg(randState) / (float)0x01000000;
}

__device__ __forceinline__ float2 rect_lerp(const Rect &rect,
                                            float u,
                                            float v) {
  return make_float2(rect.x + rect.width * u, rect.y + rect.height * v);
}

__device__ __forceinline__ uint2 current_pixel() {
  auto launchIndex = optixGetLaunchIndex();
  const auto &frame = g_LaunchParams.outputFrame;
  auto pixel =
      make_uint2(launchIndex.x + frame.tile.x, launchIndex.y + frame.tile.y);
  return pixel;
}

__device__ __forceinline__ float3 &pixel_value(const uint2 &pixel) {
  auto index = pixel.y * g_LaunchParams.outputFrame.width + pixel.x;
  return g_LaunchParams.outputFrame.buffer[index];
}

__device__ __forceinline__ float3 &current_pixel_value() {
  return pixel_value(current_pixel());
}

__device__ __forceinline__ float
sample_camera_ray(unsigned int &randState, float3 &origin, float3 &direction) {
  const auto &cam = g_LaunchParams.camera;
  const auto &frame = g_LaunchParams.outputFrame;
  auto width = (float)frame.width;
  auto height = (float)frame.height;
  auto pixelIndex = current_pixel();
  auto pixel = make_float2(pixelIndex.x, pixelIndex.y);
  // jitter pixel position
  pixel.x += rnd(randState);
  pixel.y += rnd(randState);

  auto canvasXY = rect_lerp(cam.canvas, pixel.x / width, pixel.y / height);
  auto dir = cam.right * canvasXY.x + cam.up * canvasXY.y - cam.back;

  origin = cam.position;
  direction = normalize(dir);
  return 1.0f; // w / pdf
}

struct MaterialInfo {
  float3 baseColor;
  float3 emission;
};

__device__ __forceinline__ void init_material_info(MaterialInfo &mat) {
  mat.baseColor = make_float3(0, 0, 0);
  mat.emission = make_float3(0, 0, 0);
}

struct TangentSpace {
  float3 origin;
  float3 n;
  float3 dpdu;
  float3 dpdv;
};

__device__ __forceinline__ void init_tangent_space(TangentSpace &ts) {
  ts.origin = make_float3(0, 0, 0);
  ts.n = make_float3(0, 0, 0);
  ts.dpdu = make_float3(0, 0, 0);
  ts.dpdv = make_float3(0, 0, 0);
}

struct GeometryInfo {
  unsigned int primId;
  float3 vertices[3];
  float2 uv;
  TangentSpace ts;
};

__device__ __forceinline__ void init_geometry_info(GeometryInfo &geom) {
  geom.primId = 0;
  for (int i = 0; i < 3; i++) {
    geom.vertices[i] = make_float3(0, 0, 0);
  }
  geom.uv = make_float2(0, 0);
  init_tangent_space(geom.ts);
}

enum class HitType { Unknown, Surface, Miss };

struct RayPayload {
  HitType hitType;
  MaterialInfo mat;
  GeometryInfo geom;
};

__device__ __forceinline__ void init_prd(RayPayload &prd) {
  prd.hitType = HitType::Unknown;
  init_material_info(prd.mat);
  init_geometry_info(prd.geom);
}

__device__ __forceinline__ void
unpack_ptr(void *ptr, unsigned int &p0, unsigned int &p1) {
  p0 = ((size_t)ptr) >> 32;
  p1 = ((size_t)ptr) & (unsigned int)(-1);
}

__device__ __forceinline__ void *pack_ptr(unsigned int p0, unsigned int p1) {
  size_t p = (((size_t)p0) << 32) | (size_t)p1;
  return (void *)p;
}

__device__ __forceinline__ RayPayload *get_prd() {
  return (RayPayload *)pack_ptr(optixGetPayload_0(), optixGetPayload_1());
}

// z goes upward
__device__ __forceinline__ float
uniform_sample_hemisphere(unsigned int &randState, float3 &d) {
  auto theta = 2.0f * Pi * rnd(randState);
  auto z = rnd(randState);
  auto r = sqrtf(max(0.0f, 1.0f - z * z));
  d.x = r * cosf(theta);
  d.y = r * sinf(theta);
  d.z = z;
  return 0.5f * InvPi;
}

// z goes upward
__device__ __forceinline__ float
cosine_sample_hemisphere(unsigned int &randState, float3 &d) {
  auto theta = 2.0f * Pi * rnd(randState);
  auto r = sqrtf(rnd(randState));
  auto z = sqrtf(1 - r * r);
  d.x = r * cosf(theta);
  d.y = r * sinf(theta);
  d.z = z;
  return z * InvPi;
}

extern "C" __device__ void __raygen__entry() {
  auto pixelIndex = current_pixel();
  unsigned int randState = pixelIndex.x * 114514 + pixelIndex.y;

  const auto spp = g_LaunchParams.spp;
  auto maxDepth = g_LaunchParams.maxDepth;

  float3 pixelColor = make_float3(0, 0, 0);

  for (int i = 0; i < spp; i++) {
    float3 rayOrigin, rayDirection;
    auto w = sample_camera_ray(randState, rayOrigin, rayDirection);

    float3 rayColor = make_float3(0, 0, 0);
    float3 factor = make_float3(1.0f, 1.0f, 1.0f);

    const auto &scene = g_LaunchParams.scene;

    for (int depth = 0; depth < maxDepth; depth++) {
      RayPayload prd{};
      init_prd(prd);
      unsigned int p0, p1;
      unpack_ptr(&prd, p0, p1);
      optixTrace(scene.gas,
                 rayOrigin,
                 rayDirection,
                 scene.epsilon,          // tmin
                 max(scene.extent, 1e4), // tmax
                 0,                      // ray time
                 255,                    // mask
                 OPTIX_RAY_FLAG_NONE,
                 0, // sbt offset
                 1, // sbt stride
                 0, // miss index
                 p0,
                 p1);
      if (prd.hitType == HitType::Miss) {
        rayColor += factor * prd.mat.emission;
        break;
      } else if (prd.hitType == HitType::Surface) {
        rayColor += factor * prd.mat.emission;
        // calculate next ray
        float3 d;
        auto pdf = cosine_sample_hemisphere(randState, d);
        const auto &ts = prd.geom.ts;
        float3 localX = normalize(ts.dpdu);
        float3 localZ = faceforward(ts.n, -rayDirection, ts.n);
        float3 localY = normalize(cross(ts.n, localX));
        auto nextDir = normalize(localX * d.x + localY * d.y + localZ * d.z);
        // init next ray
        rayOrigin = ts.origin + localZ * scene.epsilon;
        rayDirection = nextDir;
        // attenuate factor, some terms are cancelled out as
        // hemisphere are cosine sampled
        factor *= prd.mat.baseColor;
      } else {
        // unknow hit type
        optixThrowException(123);
      }
    }
    pixelColor += w * rayColor;
  }
  current_pixel_value() = pixelColor / (float)spp;
}

extern "C" __device__ void __miss__entry() {
  auto data = (MissData *)optixGetSbtDataPointer();
  auto prd = get_prd();

  prd->hitType = HitType::Miss;
  prd->mat.baseColor = data->color;
}

extern "C" __device__ void __exception__entry() {
  auto data = (ExceptionData *)optixGetSbtDataPointer();
  current_pixel_value() = data->errorColor;
}

extern "C" __device__ void __closesthit__entry() {
  auto prd = get_prd();
  prd->hitType = HitType::Surface;

  // init geom
  auto data = (HitGroupData *)optixGetSbtDataPointer();
  auto primId = optixGetPrimitiveIndex();
  auto gas = optixGetGASTraversableHandle();
  auto uv = optixGetTriangleBarycentrics();
  prd->geom.primId = primId;
  prd->geom.uv = uv;
  optixGetTriangleVertexData(gas, primId, 0, 0, prd->geom.vertices);

  // calculate tangent space
  auto &ts = prd->geom.ts;
  auto v = prd->geom.vertices;
  ts.origin = v[0] * (1 - uv.x - uv.y) + v[1] * uv.x + v[2] * uv.y;
  ts.dpdu = v[1] - v[0];
  ts.dpdv = v[2] - v[0];
  ts.n = normalize(cross(ts.dpdu, ts.dpdv));

  // init material
  prd->mat.baseColor = data->baseColor;
  prd->mat.emission = data->emission;
}

extern "C" __device__ void __anyhit__entry() {}