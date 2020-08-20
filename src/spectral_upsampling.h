#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector_types.h>

struct RGB2Spectral {
  uint32_t resolution;
  std::unique_ptr<float[]> scales;
  std::unique_ptr<float[]> data;

  static std::unique_ptr<RGB2Spectral> load(const std::string &path);
  float3 rgb_to_coeff(const float3 &rgb) const;
};

