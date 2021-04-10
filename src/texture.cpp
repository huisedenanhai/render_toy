#include "texture.h"
#include "exceptions.h"
#include "scene.h"
#include <cuda_runtime.h>
#include <sstream>
#include <stb/stb_image.h>
#include <stdexcept>
#include <vector>

void Texture::destroy() {
  cudaDestroyTextureObject(tex);
  cudaFreeArray(pixelArray);
  tex = 0;
  pixelArray = nullptr;
}

namespace {
std::vector<float> process_data(uint8_t *data,
                                int32_t width,
                                int32_t height,
                                int32_t numComponents,
                                int options) {
  std::vector<float> pixels;
  pixels.resize(4 * width * height);
  for (int i = 0; i < width * height; i++) {
    for (int c = 0; c < 4; c++) {
      auto index = i * 4 + c;
      if (c >= numComponents) {
        pixels[index] = 0.0f;
      } else {
        auto v = (float)(data[i * numComponents + c]) / 255.0f;
        if (options & TextureLoader::SRGB_TO_LINEAR) {
          v = powf(v, 2.2f);
        }
        pixels[index] = v;
      }
    }
  }

  if (options & TextureLoader::RGB_TO_SPECTRAL) {
    for (int i = 0; i < width * height; i++) {
      auto offset = i * 4;
      auto rgb2spec = Scene::rgb2spectral.get();
      auto coeff = rgb2spec->rgb_to_coeff(
          make_float3(pixels[offset], pixels[offset + 1], pixels[offset + 2]));
      pixels[offset] = coeff.x;
      pixels[offset + 1] = coeff.y;
      pixels[offset + 2] = coeff.z;
    }
  }
  return pixels;
}
} // namespace

Texture TextureLoader::load(const char *path, int options) {
  int width, height, numComponents;
  unsigned char *data = stbi_load(path, &width, &height, &numComponents, 0);
  if (!data) {
    std::stringstream ss;
    ss << "failed to load image " << path;
    throw std::runtime_error(ss.str());
  }

  auto pixels = process_data(data, width, height, numComponents, options);

  stbi_image_free(data);

  cudaChannelFormatDesc channelDesc;
  int32_t pitch = width * 4 * sizeof(float);
  channelDesc = cudaCreateChannelDesc<float4>();

  cudaArray_t pixelArray;
  TOY_CUDA_CHECK_OR_THROW(
      cudaMallocArray(&pixelArray, &channelDesc, width, height), );

  TOY_CUDA_CHECK_OR_THROW(cudaMemcpy2DToArray(pixelArray,
                                              /* offset */ 0,
                                              0,
                                              pixels.data(),
                                              pitch,
                                              pitch,
                                              height,
                                              cudaMemcpyHostToDevice), );

  cudaResourceDesc res_desc = {};
  res_desc.resType = cudaResourceTypeArray;
  res_desc.res.array.array = pixelArray;

  cudaTextureDesc tex_desc = {};
  tex_desc.addressMode[0] = cudaAddressModeWrap;
  tex_desc.addressMode[1] = cudaAddressModeWrap;
  tex_desc.filterMode = cudaFilterModeLinear;
  tex_desc.readMode = cudaReadModeElementType;
  tex_desc.normalizedCoords = 1;
  tex_desc.maxAnisotropy = 1;
  tex_desc.maxMipmapLevelClamp = 99;
  tex_desc.minMipmapLevelClamp = 0;
  tex_desc.mipmapFilterMode = cudaFilterModePoint;
  tex_desc.borderColor[0] = 1.0f;
  tex_desc.sRGB = 0;

  // Create texture object
  cudaTextureObject_t cuda_tex = 0;
  TOY_CUDA_CHECK_OR_THROW(
      cudaCreateTextureObject(&cuda_tex, &res_desc, &tex_desc, nullptr), );

  Texture result{};
  result.pixelArray = pixelArray;
  result.tex = cuda_tex;
  return result;
}