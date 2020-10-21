#include "cmf.h"
#include "d65.h"
#include "pipeline.h"
#include "random.h"
#include "vec_math.h"
#include <cstdint>
#include <optix.h>

using namespace dev;

// #define USE_CMF_IMPORTANCE_SAMPLING

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

__device__ __forceinline__ uint32_t reverse_bits_32(uint32_t n) {
  n = (n << 16) | (n >> 16);
  n = ((n & 0x00ff00ff) << 8) | ((n & 0xff00ff00) >> 8);
  n = ((n & 0x0f0f0f0f) << 4) | ((n & 0xf0f0f0f0) >> 4);
  n = ((n & 0x33333333) << 2) | ((n & 0xcccccccc) >> 2);
  n = ((n & 0x55555555) << 1) | ((n & 0xaaaaaaaa) >> 1);
  return n;
}

__device__ __forceinline__ float radical_inverse_base_2(uint32_t n) {
  return saturate(reverse_bits_32(n) * float(2.3283064365386963e-10));
}

// 0 < i <= n
__device__ __forceinline__ float2 hammersley_sample(uint32_t i, uint32_t n) {
  return make_float2((float)(i) / (float)(n), radical_inverse_base_2(i));
}

// 0 < i <= spp
__device__ __forceinline__ float
sample_camera_ray(uint32_t n, uint32_t spp, float3 &origin, float3 &direction) {
  const auto &cam = g_LaunchParams.camera;
  const auto &frame = g_LaunchParams.outputFrame;
  auto width = (float)frame.width;
  auto height = (float)frame.height;
  auto pixelIndex = current_pixel();
  auto pixel = make_float2(pixelIndex.x, pixelIndex.y);
  // jitter pixel position
  auto sample = hammersley_sample(n, spp);
  pixel.x += sample.x;
  pixel.y += sample.y;

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
  RayType type;
  int length;
  float4 lambda;
  float4 waveLength;
  float4 weight;
  float4 pdf;
  float4 L;
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
  auto y = sqrtf(max(1 - r * r, 0.0f));
  d.x = r * cosf(theta);
  d.y = y;
  d.z = r * sinf(theta);
  return y * InvPi;
}

__device__ __forceinline__ float lerp(float a, float b, float t) {
  return a + t * (b - a);
}

__device__ __forceinline__ float4 lerp(float4 a, float4 b, float4 t) {
  return make_float4(lerp(a.x, b.x, t.x),
                     lerp(a.y, b.y, t.y),
                     lerp(a.z, b.z, t.z),
                     lerp(a.w, b.w, t.w));
}

__device__ __forceinline__ float4 lerp(float a, float b, float4 t) {
  return lerp(make_float4(a, a, a, a), make_float4(b, b, b, b), t);
}

__device__ __forceinline__ float inverse_lerp(float a, float b, float v) {
  return (v - a) / (b - a);
}

__device__ __forceinline__ float clamp(float v, float a, float b) {
  return max(min(v, b), a);
}

__device__ __forceinline__ float
sample_array(float *data, uint32_t count, float u) {
  uint32_t i = clamp(count * u, 0, count - 2);
  return lerp(data[i], data[i + 1], saturate(count * u - i));
}

__device__ __forceinline__ float3 sample_xyz(float lambda) {
  auto sample_comp = [&](float *cmf) {
    return sample_array(cmf, CMF_WaveLengthCount, lambda);
  };
  return make_float3(
      sample_comp(CMF_X), sample_comp(CMF_Y), sample_comp(CMF_Z));
}

// sample two iid normal dist values from two uniform samples in range(0, 1)
// http://bjlkeng.github.io/posts/sampling-from-a-normal-distribution/
__device__ __forceinline__ float2 box_muller(float u, float v) {
  float r = sqrtf(max(-2.0f * logf(max(1e-9f, u)), 0.0f));
  float t = 2.0f * Pi * v;
  return make_float2(r * cosf(t), r * sinf(t));
}

// map a value from N(0, 1) to N(mean, stdvar)
// distribution
__device__ __forceinline__ float
normal_dist_remap(float v, float mean, float stdvar) {
  return v * stdvar + mean;
}

__device__ __forceinline__ float
normal_dist_pdf(float v, float mean, float stdvar) {
  float vNorm = (v - mean) / stdvar;
  return expf(-vNorm * vNorm * 0.5f) / sqrtf(2.0f * Pi) / stdvar;
}

__device__ __forceinline__ void
sample_wave_length(unsigned int &seed, float lambda[4], float pdf[4]) {
#ifdef USE_CMF_IMPORTANCE_SAMPLING
  float normalDist[4];
  for (int i = 0; i < 4; i += 2) {
    auto ns = box_muller(rnd(seed), rnd(seed));
    normalDist[i] = ns.x;
    normalDist[i + 1] = ns.y;
  }
  for (int i = 0; i < 4; i++) {
    float rate = CMF_FitA1 / (CMF_FitA1 + CMF_FitA2);
    float choose1 = rnd(seed) < rate;
    float average = choose1 ? CMF_FitMean1 : CMF_FitMean2;
    float stdvar = choose1 ? CMF_FitStdVar1 : CMF_FitStdVar2;
    float l = saturate(normal_dist_remap(normalDist[i], average, stdvar));
    lambda[i] = l;
    pdf[i] = normal_dist_pdf(l, CMF_FitMean1, CMF_FitStdVar1) * CMF_FitA1 +
             normal_dist_pdf(l, CMF_FitMean2, CMF_FitStdVar2) * CMF_FitA2;
  }
#else
  for (int i = 0; i < 4; i++) {
    lambda[i] = rnd(seed);
    pdf[i] = 1.0f;
  }
#endif
}

__device__ __forceinline__ float eval_spectrum(const float3 &coeff,
                                               float waveLength) {
  return eval_rgb_to_spectral_coeff(coeff, waveLength);
}

__device__ __forceinline__ float4 eval_spectrum(const float3 &coeff,
                                                float4 waveLength) {
  return make_float4(eval_rgb_to_spectral_coeff(coeff, waveLength.x),
                     eval_rgb_to_spectral_coeff(coeff, waveLength.y),
                     eval_rgb_to_spectral_coeff(coeff, waveLength.z),
                     eval_rgb_to_spectral_coeff(coeff, waveLength.w));
}

extern "C" __device__ void __raygen__entry() {
  auto pixelIndex = current_pixel();
  RayPayload prd{};
  prd.seed = tea<4>(pixel_index(pixelIndex), 114514);

  unsigned int p0, p1;
  unpack_ptr(&prd, p0, p1);

  float3 pixelColorXYZ = make_float3(0.0f, 0.0f, 0.0f);
  auto spp = g_LaunchParams.spp;

  for (int i = 0; i < spp; i++) {
    prd.finish = false;
    prd.length = 0;
    prd.type = RayTypeCamera;
    float waveLengthLambda[4];
    float waveLengthPDF[4];
    sample_wave_length(prd.seed, waveLengthLambda, waveLengthPDF);
    prd.lambda = make_float4(waveLengthLambda[0],
                             waveLengthLambda[1],
                             waveLengthLambda[2],
                             waveLengthLambda[3]);
    prd.waveLength = lerp(CMF_MinWaveLength, CMF_MaxWaveLength, prd.lambda);
    prd.weight = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
    prd.pdf = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
    prd.L = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    auto w = sample_camera_ray(i + 1, spp, prd.ray.origin, prd.ray.direction);
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
    // CMIS
    auto invSumPdf =
        1.0f / max(prd.pdf.x + prd.pdf.y + prd.pdf.z + prd.pdf.w, 1e-7f);
    pixelColorXYZ += w * prd.pdf.x * invSumPdf * prd.L.x *
                     sample_xyz(prd.lambda.x) / waveLengthPDF[0];
    pixelColorXYZ += w * prd.pdf.y * invSumPdf * prd.L.y *
                     sample_xyz(prd.lambda.y) / waveLengthPDF[1];
    pixelColorXYZ += w * prd.pdf.z * invSumPdf * prd.L.z *
                     sample_xyz(prd.lambda.z) / waveLengthPDF[2];
    pixelColorXYZ += w * prd.pdf.w * invSumPdf * prd.L.w *
                     sample_xyz(prd.lambda.w) / waveLengthPDF[3];
  }
  pixelColorXYZ /= (float)spp;
  current_pixel_value() = xyz_to_srgb(pixelColorXYZ);
}

extern "C" __device__ void __miss__entry() {
  auto data = (MissData *)optixGetSbtDataPointer();
  auto prd = get_prd();

  // prd->L += prd->weight * eval_spectrum(data->colorCoeff, prd->waveLength);
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
  bool enableRR = prd->length >= 3;
  auto continueRate =
      enableRR ? min(max(length(prd->weight), 0.03f), 1.0f) : 1.0f;
  bool rrFinish = enableRR && rnd(prd->seed) > continueRate;
  prd->finish = prd->length >= g_LaunchParams.maxPathLength || rrFinish;
  prd->weight /= continueRate;
}

extern "C" __device__ void __closesthit__blackbody() {
  auto prd = get_prd();
  auto data = (BlackBodyHitGroupData *)optixGetSbtDataPointer();

  prd->L += prd->weight *
            make_float4(data->sample_spectrum_scaled(prd->waveLength.x),
                        data->sample_spectrum_scaled(prd->waveLength.y),
                        data->sample_spectrum_scaled(prd->waveLength.z),
                        data->sample_spectrum_scaled(prd->waveLength.w));
  // prd->L +=
  //     prd->weight *
  //     sample_array(
  //         D65_SPD,
  //         D65_WaveLengthCount,
  //         inverse_lerp(D65_MinWaveLength, D65_MaxWaveLength,
  //         prd->waveLength)) *
  //     100;
  prd->finish = true;
}

extern "C" __device__ void __anyhit__blackbody() {}

extern "C" __device__ void __closesthit__diffuse() {
  auto prd = get_prd();
  auto data = (DiffuseHitGroupData *)optixGetSbtDataPointer();
  auto geom = get_geometry();
  TangentSpace &ts = geom.ts;

  // prd->L += prd->weight * eval_spectrum(data->emissionCoeff,
  // prd->waveLength);
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
  prd->weight *= eval_spectrum(data->baseColorCoeff, prd->waveLength);
  prd->pdf *= pdf;
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
  // only consider hero ray
  float ior = data->get_ior(prd->waveLength.x);
  float nT = wi.y > 0 ? ior : 1.0f / ior;
  float nI = 1.0f / nT;
  float cosI = abs(wi.y);
  float sinI = sqrtf(max(1.0f - cosI * cosI, 0.00001f));
  float sinT = sinI * nI;
  float cosT = sqrtf(max(1.0f - sinT * sinT, 0.00001f));
  auto fr = fresnel(cosI, cosT, nT);
  auto fReflect = fr;
  // we have fresnel(cosT, cosI, nI) == fresnel(cosI, cosT, nT)
  auto fRefract = (1.0f - fr) * nI * nI;
  auto reflectRate = sinT > 1.0f ? 1.0f : fReflect / (fReflect + fRefract);
  auto reflectDirLocal = make_float3(-wi.x, wi.y, -wi.z);
  auto invLenXZ = 1.0f / max(sqrtf(wi.x * wi.x + wi.z * wi.z), 0.00001f);
  auto refractDirLocal = normalize(make_float3(-wi.x * sinT * invLenXZ,
                                               copysignf(cosT, -wi.y),
                                               -wi.z * sinT * invLenXZ));

  bool doReflect = rnd(prd->seed) <= reflectRate;
  auto nextDirLocal = doReflect ? reflectDirLocal : refractDirLocal;
  float pdf = doReflect ? reflectRate : 1.0f - reflectRate;

  auto nextDir = base.local_to_world_dir(nextDirLocal);
  // init next ray
  auto &scene = g_LaunchParams.scene;
  prd->ray.origin =
      ts.origin + faceforward(ts.n, nextDir, ts.n) * scene.epsilon;
  prd->ray.direction = nextDir;
  prd->ray.min = scene.epsilon;
  prd->ray.max = scene.extent;
  if (data->cauchy != 0) {
    // other wavelength has no contribution when dispersion happens
    prd->weight *=
        make_float4(eval_spectrum(data->baseColorCoeff, prd->waveLength.x),
                    0.0f,
                    0.0f,
                    0.0f);
    prd->pdf *= make_float4(pdf, 0.0f, 0.0f, 0.0f);
  } else {
    // no dispersion
    prd->weight *= eval_spectrum(data->baseColorCoeff, prd->waveLength);
    prd->pdf *= pdf;
  }
}

extern "C" __device__ void __anyhit__glass() {}