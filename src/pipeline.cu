#include "pipeline.h"
#include "random.h"
#include "vec_math.h"
#include <optix.h>

using namespace dev;

extern "C" {
__constant__ LaunchParams g_LaunchParams;
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

__device__ __forceinline__ unsigned int pixel_index(const uint2 &pixel) {
  return pixel.y * g_LaunchParams.outputFrame.width + pixel.x;
}

__device__ __forceinline__ float3 &pixel_value(const uint2 &pixel) {
  auto index = pixel_index(pixel);
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

// forms a xyz right hand coord
struct OthoBase {
  float3 x;
  float3 y;
  float3 z;

  __device__ __forceinline__ float3 local_to_world_dir(float3 d) {
    return x * d.x + y * d.y + z * d.z;
  }

  __device__ __forceinline__ float3 world_to_local_dir(float3 d) {
    return make_float3(dot(d, x), dot(d, y), dot(d, z));
  }
};

struct TangentSpace {
  float3 origin;
  float3 dpdu;
  float3 dpdv;
  float3 n;

  // n aligns with y
  // assume n is normalized, and perpendicular to both dpdu and dpdv
  __device__ __forceinline__ OthoBase get_otho_base() {
    OthoBase base;
    base.x = normalize(dpdu);
    base.y = n;
    base.z = cross(base.x, base.y);
    return base;
  }
};

struct Ray {
  float3 origin;
  float min;
  float3 direction;
  float max;
};

struct RayPayload {
  bool finish;
  Ray ray;
  int length;
  float3 weight;
  float3 color;
  unsigned int seed;
};

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

// y goes upward
__device__ __forceinline__ float
uniform_sample_hemisphere(unsigned int &randState, float3 &d) {
  auto theta = 2.0f * Pi * rnd(randState);
  auto y = rnd(randState);
  auto r = sqrtf(max(0.0f, 1.0f - y * y));
  d.x = r * cosf(theta);
  d.y = y;
  d.z = r * sinf(theta);
  return 0.5f * InvPi;
}

// y goes upward
__device__ __forceinline__ float
cosine_sample_hemisphere(unsigned int &randState, float3 &d) {
  auto theta = 2.0f * Pi * rnd(randState);
  auto r = sqrtf(rnd(randState));
  auto y = sqrtf(1 - r * r);
  d.x = r * cosf(theta);
  d.y = y;
  d.z = r * sinf(theta);
  return y * InvPi;
}

extern "C" __device__ void __raygen__entry() {
  auto pixelIndex = current_pixel();
  RayPayload prd{};
  prd.seed = tea<4>(pixel_index(pixelIndex), 114514);

  unsigned int p0, p1;
  unpack_ptr(&prd, p0, p1);

  float3 pixelColor = make_float3(0.0f, 0.0f, 0.0f);
  auto spp = g_LaunchParams.spp;

  for (int i = 0; i < spp; i++) {
    prd.finish = false;
    prd.length = 0;
    prd.weight = make_float3(1.0f, 1.0f, 1.0f);
    prd.color = make_float3(0.0f, 0.0f, 0.0f);

    auto w = sample_camera_ray(prd.seed, prd.ray.origin, prd.ray.direction);
    const auto &scene = g_LaunchParams.scene;
    prd.ray.min = scene.epsilon;
    prd.ray.max = scene.extent;

    while (!prd.finish) {
      prd.length++;
      optixTrace(scene.gas,
                 prd.ray.origin,
                 prd.ray.direction,
                 prd.ray.min, // tmin
                 prd.ray.max, // tmax
                 0,           // ray time
                 255,         // mask
                 OPTIX_RAY_FLAG_NONE,
                 0, // sbt offset
                 1, // sbt stride
                 0, // miss index
                 p0,
                 p1);
    }
    pixelColor += w * prd.color;
  }
  current_pixel_value() = pixelColor / (float)spp;
}

extern "C" __device__ void __miss__entry() {
  auto data = (MissData *)optixGetSbtDataPointer();
  auto prd = get_prd();

  prd->color += prd->weight * data->color;
  prd->finish = true;
}

extern "C" __device__ void __exception__entry() {
  auto data = (ExceptionData *)optixGetSbtDataPointer();
  current_pixel_value() = data->errorColor;
}

struct Geometry {
  unsigned int primId;
  float2 uv;
  float3 v[3];
  TangentSpace ts;
};

__device__ __forceinline__ Geometry get_geometry() {
  Geometry geom;
  // init geom
  auto primId = optixGetPrimitiveIndex();
  auto gas = optixGetGASTraversableHandle();
  auto uv = optixGetTriangleBarycentrics();
  optixGetTriangleVertexData(gas, primId, 0, 0, geom.v);
  geom.primId = primId;
  geom.uv = uv;
  float3 *v = geom.v;

  // calculate tangent space
  TangentSpace &ts = geom.ts;
  ts.origin = v[0] * (1 - uv.x - uv.y) + v[1] * uv.x + v[2] * uv.y;
  ts.dpdu = v[1] - v[0];
  ts.dpdv = v[2] - v[0];
  ts.n = normalize(cross(ts.dpdu, ts.dpdv));
  return geom;
}

__device__ __forceinline__ void russian_roulette(RayPayload *prd) {
  // Russian Roulette
  // don't make the prob too small
  auto continueRate = min(max(length(prd->weight), 0.03f), 1.0f);
  bool rrFinish = prd->length >= 3 && rnd(prd->seed) > continueRate;
  prd->finish = prd->length >= g_LaunchParams.maxPathLength || rrFinish;
  prd->weight /= continueRate;
}

extern "C" __device__ void __closesthit__diffuse() {
  auto prd = get_prd();
  auto data = (DiffuseHitGroupData *)optixGetSbtDataPointer();
  auto geom = get_geometry();
  TangentSpace &ts = geom.ts;

  prd->color += prd->weight * data->emission;
  // rr will modify the weight
  russian_roulette(prd);

  // calculate next ray
  float3 d;
  auto pdf = cosine_sample_hemisphere(prd->seed, d);
  auto base = ts.get_otho_base();
  base.y = faceforward(base.y, -prd->ray.direction, base.y);
  auto nextDir = base.local_to_world_dir(d);
  // init next ray
  auto &scene = g_LaunchParams.scene;
  prd->ray.origin = ts.origin + base.y * scene.epsilon;
  prd->ray.direction = nextDir;
  prd->ray.min = scene.epsilon;
  prd->ray.max = scene.extent;
  // attenuate factor, some terms are cancelled out as
  // hemisphere are cosine sampled
  prd->weight *= data->baseColor;
}

extern "C" __device__ void __anyhit__diffuse() {}

__device__ __forceinline__ float
fresnel_parrallel(float cosI, float cosT, float nT) {
  auto r = (nT * cosI - cosT) / (nT * cosI + cosT);
  return r * r;
}

__device__ __forceinline__ float
fresnel_perpendicular(float cosI, float cosT, float nT) {
  auto r = (cosI - nT * cosT) / (cosI + nT * cosT);
  return r * r;
}

__device__ __forceinline__ float fresnel(float cosI, float cosT, float nT) {
  return 0.5f * (fresnel_parrallel(cosI, cosT, nT) +
                 fresnel_perpendicular(cosI, cosT, nT));
}

extern "C" __device__ void __closesthit__glass() {
  auto prd = get_prd();
  auto data = (GlassHitGroupData *)optixGetSbtDataPointer();
  auto geom = get_geometry();
  TangentSpace &ts = geom.ts;

  russian_roulette(prd);

  auto base = ts.get_otho_base();
  float3 wi = base.world_to_local_dir(-prd->ray.direction);
  float nT = wi.y > 0 ? data->ior : 1.0f / data->ior;
  float cosI = abs(wi.y);
  float sinI = sqrtf(max(1.0f - cosI * cosI, 0.00001f));
  float sinT = sinI / nT;
  float cosT = sqrtf(max(1.0f - sinT * sinT, 0.00001f));
  auto fr = fresnel(cosI, cosT, nT);
  auto reflectRate = sinT > 1.0f ? 1.2f : fr;
  auto reflectDirLocal = make_float3(-wi.x, wi.y, -wi.z);
  auto refractDirLocal = normalize(make_float3(-wi.x / nT, -wi.y, -wi.z / nT));
  auto nextDirLocal =
      rnd(prd->seed) < reflectRate ? reflectDirLocal : refractDirLocal;
  auto nextDir = base.local_to_world_dir(nextDirLocal);
  // init next ray
  auto &scene = g_LaunchParams.scene;
  prd->ray.origin =
      ts.origin + faceforward(ts.n, nextDir, ts.n) * scene.epsilon;
  prd->ray.direction = nextDir;
  prd->ray.min = scene.epsilon;
  prd->ray.max = scene.extent;
  prd->weight *= data->baseColor;
}

extern "C" __device__ void __anyhit__glass() {}