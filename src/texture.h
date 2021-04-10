#pragma once

#include "spectral_upsampling.h"
#include <cuda_runtime.h>

struct Texture {
  cudaArray_t pixelArray{};
  cudaTextureObject_t tex{};

  void destroy();
};

struct TextureLoader {
  enum Option {
    SRGB_TO_LINEAR = 1,
    RGB_TO_SPECTRAL = 2,
  };

  Texture load(const char *path, int options);
};