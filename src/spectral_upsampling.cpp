#include "spectral_upsampling.h"
#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <vector_functions.h>

#define SPEC_N_COEFF 3

std::unique_ptr<RGB2Spectral> RGB2Spectral::load(const std::string &path) {
  std::ifstream is(path, std::ios::binary);
  if (!is.good()) {
    throw std::runtime_error("can not open rgb to spectral data " + path);
  }
  char header[4];
  if (!is.read(header, 4) || memcmp(header, "SPEC", 4) != 0) {
    throw std::runtime_error("invalid rgb2spectral coeff file header " + path);
  }
  auto model = std::make_unique<RGB2Spectral>();
  if (!is.read((char *)&model->resolution, sizeof(uint32_t))) {
    return nullptr;
  }
  auto scaleSize = model->resolution;
  auto dataSize = 3 * model->resolution * model->resolution *
                  model->resolution * SPEC_N_COEFF;

  model->scales = std::make_unique<float[]>(scaleSize);
  model->data = std::make_unique<float[]>(dataSize);
  if (!is.read((char *)model->scales.get(), sizeof(float) * scaleSize)) {
    throw std::runtime_error("failed to read scales " + path);
  }
  if (!is.read((char *)model->data.get(), sizeof(float) * dataSize)) {
    throw std::runtime_error("failed to read data " + path);
  }
  return model;
}

inline uint32_t binary_search(const float *data, uint32_t size, float v) {
  uint32_t i = 0;
  uint32_t j = size - 1;

  while (i < j) {
    uint32_t m = (i + j) / 2;
    if (v > data[m + 1]) {
      i = m + 1;
    } else if (v < data[m]) {
      j = m;
    } else {
      return m;
    }
  }
  return std::min(i, size - 1);
}

inline float lerp(float a, float b, float t) {
  return a + t * (b - a);
}

float3 RGB2Spectral::rgb_to_coeff(const float3 &rgb) const {
  assert(rgb.x >= 0.0f && rgb.y >= 0.0f && rgb.z >= 0.0f);
  assert(rgb.x <= 1.0f && rgb.y <= 1.0f && rgb.z <= 1.0f);
  if (rgb.x == 0 && rgb.y == 0 && rgb.z == 0) {
    return make_float3(0, 0, 0);
  }

  float rgbValues[3] = {rgb.x, rgb.y, rgb.z};
  int maxCompIndex = 0;
  for (int i = 0; i < 3; i++) {
    if (rgbValues[i] > rgbValues[maxCompIndex]) {
      maxCompIndex = i;
    }
  }

  float z = rgbValues[maxCompIndex];
  float s = (resolution - 1) / z;
  float x = rgbValues[(maxCompIndex + 1) % 3] * s;
  float y = rgbValues[(maxCompIndex + 2) % 3] * s;

  uint32_t xi = std::min((uint32_t)x, resolution - 2);
  uint32_t yi = std::min((uint32_t)y, resolution - 2);
  uint32_t zi =
      std::min(binary_search(scales.get(), resolution, z), resolution - 2);
  float xt = x - xi;
  float yt = y - yi;
  float zt = (z - scales[zi]) / (scales[zi + 1] - scales[zi]);

  uint32_t dx = SPEC_N_COEFF;
  uint32_t dy = SPEC_N_COEFF * resolution;
  uint32_t dz = SPEC_N_COEFF * resolution * resolution;
  uint32_t offset =
      (zi * dz + yi * dy + xi * dx) +
      maxCompIndex * SPEC_N_COEFF * resolution * resolution * resolution;

  float coeffValues[SPEC_N_COEFF];
  for (int i = 0; i < SPEC_N_COEFF; i++) {
    coeffValues[i] =
        lerp(lerp(lerp(data[offset], data[offset + dx], xt),
                  lerp(data[offset + dy], data[offset + dx + dy], xt),
                  yt),
             lerp(lerp(data[offset + dz], data[offset + dx + dz], xt),
                  lerp(data[offset + dy + dz], data[offset + dx + dy + dz], xt),
                  yt),
             zt);
    offset++;
  }
  return make_float3(coeffValues[0], coeffValues[1], coeffValues[2]);
}