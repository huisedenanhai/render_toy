#include "Scene.h"
#include "config.h"
#include "context.h"
#include "exceptions.h"
#include "path_tracing.h"
#include "pipeline.h"
#include "vec_math.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <ctime>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <stb/stb_image_write.h>
#include <tinyobj/tiny_obj_loader.h>
#include <unordered_map>
#include <vector>

static void configure_ptx_dir(const char *exeDir) {
  int pathLen = parent_dir_len(exeDir);
  char seperator = exeDir[pathLen];
  {
    auto buf = std::make_unique<char[]>(pathLen + 10);
    memcpy(buf.get(), exeDir, pathLen);
    buf[pathLen] = seperator;
    memcpy(buf.get() + pathLen + 1, "ptx", 4);
    Context::ptxDir = buf.get();
  }

  convert_path_seperator(Context::ptxDir);
  std::cout << "compiled ptx will be load from " << Context::ptxDir
            << std::endl;
}

int run(int argc, const char **argv) {
  printf("exe: %s\n", argv[0]);
  configure_ptx_dir(argv[0]);
  if (argc < 2) {
    printf("no input file specified, stop.");
    return 0;
  }
  std::string inputFile = argv[1];
  // init optix
  Context::init();
  Scene::rgb2spectral = RGB2Spectral::load(SRC_DIR "/data/jakob_srgb.coeff");
  auto integrator = PathIntegratorBuilder().build();
  auto scene = SceneLoader().load(inputFile, *integrator);

  unsigned int frameWidth = scene.frame.width, frameHeight = scene.frame.height;
  unsigned int tileWidth = scene.tile.width, tileHeight = scene.tile.height;
  float frameResolution = scene.frame.resolution;
  unsigned int spp = scene.launch.spp;
  unsigned int maxPathLength = scene.launch.maxPathLength;
  auto outputBufferSize = frameWidth * frameHeight * sizeof(float3);
  constexpr unsigned int numStreams = 8;

  cudaStream_t streams[numStreams]{};
  for (int i = 0; i < numStreams; i++) {
    TOY_CUDA_CHECK_OR_THROW(cudaStreamCreate(&streams[i]), );
  }

  CudaMemoryRAII outputFrameBuffer;

  // launch
  {
    // alloc output frame buffer
    TOY_CUDA_CHECK_OR_THROW(
        cudaMalloc(&outputFrameBuffer.ptr, outputBufferSize), );

    dev::LaunchParams launchParams[numStreams]{};
    // init common launchParam values
    for (int i = 0; i < numStreams; i++) {
      auto &param = launchParams[i];
      auto &camera = param.camera;
      camera.position = scene.camera.position;
      camera.right = scene.camera.right;
      camera.up = scene.camera.up;
      camera.back = scene.camera.back;
      auto canvasWidth = frameWidth / frameResolution;
      auto canvasHeight = frameHeight / frameResolution;
      camera.canvas = dev::make_rect(
          -canvasWidth / 2.0f, -canvasHeight / 2.0f, canvasWidth, canvasHeight);

      auto &frame = param.outputFrame;
      frame.width = frameWidth;
      frame.height = frameHeight;
      frame.buffer = (float3 *)outputFrameBuffer.ptr;

      param.scene.gas = scene.gas.gas;
      param.scene.epsilon = 1e-5;
      param.scene.extent =
          scene.gas.aabb.extend(camera.position).get_bouding_sphere().radius *
              2.0f +
          5.0f;

      param.spp = spp;
      param.maxPathLength = maxPathLength;
    }

    CudaMemoryRAII launchParams_d;
    auto startTime = std::chrono::system_clock::now();

    {
      auto t = std::chrono::system_clock::to_time_t(startTime);
      std::cout << "render start at " << std::ctime(&t) << std::flush;
    }
    {
      TOY_CUDA_CHECK_OR_THROW(
          cudaMalloc(&launchParams_d.ptr,
                     sizeof(dev::LaunchParams) * numStreams), );
      int streamIndex = 0;
      // tile frame for launch
      for (int x = 0; x < frameWidth; x += tileWidth) {
        for (int y = 0; y < frameHeight;
             y += tileHeight, streamIndex = (streamIndex + 1) % numStreams) {
          auto stream = streams[streamIndex];
          auto &param = launchParams[streamIndex];
          auto paramPtrD = (void *)((char *)launchParams_d.ptr +
                                    streamIndex * sizeof(dev::LaunchParams));
          // init tile
          int launchWidth = std::min(tileWidth, frameWidth - x);
          int launchHeight = std::min(tileHeight, frameHeight - y);
          auto &frame = param.outputFrame;
          frame.tile = dev::make_rect_int(x, y, launchWidth, launchHeight);

          TOY_CUDA_CHECK_OR_THROW(cudaMemcpyAsync(paramPtrD,
                                                  &param,
                                                  sizeof(dev::LaunchParams),
                                                  cudaMemcpyHostToDevice,
                                                  stream), );

          // launch size should match exactly with cropped size of the ouput
          // frame
          TOY_OPTIX_CHECK_OR_THROW(optixLaunch(integrator->pipeline.pipeline,
                                               stream,
                                               (CUdeviceptr)paramPtrD,
                                               sizeof(dev::LaunchParams),
                                               &scene.sbt.sbt,
                                               launchWidth,
                                               launchHeight,
                                               1), );
        }
      }

      for (int i = 0; i < numStreams; i++) {
        TOY_CUDA_CHECK_OR_THROW(cudaStreamSynchronize(streams[i]), );
      }

      auto endTime = std::chrono::system_clock::now();
      {
        using namespace std::chrono;
        auto t = system_clock::to_time_t(endTime);
        // TODO more pretty output
        auto s = duration<float>(endTime - startTime).count();
        auto m = floor(s / 60.0);
        s -= m * 60.0f;
        auto h = floor(m / 60.0);
        m -= h * 60.0f;
        std::cout << "render finished at " << std::ctime(&t) << std::flush;
        std::cout << "duration " << h << ":" << m << ":" << s << std::endl;
      }
    }
  }
  // denoise
  if (scene.frame.denoise) {
    OptixDenoiser denoiser;
    {
      OptixDenoiserOptions denoiserOpts{};
      denoiserOpts.inputKind = OPTIX_DENOISER_INPUT_RGB;
      TOY_OPTIX_CHECK_OR_THROW(
          optixDenoiserCreate(Context::context, &denoiserOpts, &denoiser), );
      TOY_OPTIX_CHECK_OR_THROW(
          optixDenoiserSetModel(
              denoiser, OPTIX_DENOISER_MODEL_KIND_HDR, nullptr, 0), );
    }
    OptixDenoiserSizes denoiserSizes;
    TOY_OPTIX_CHECK_OR_THROW(
        optixDenoiserComputeMemoryResources(
            denoiser, frameWidth, frameHeight, &denoiserSizes), );

    CudaMemoryRAII intensity_d{};
    CudaMemoryRAII scratch_d{};
    CudaMemoryRAII state_d{};
    size_t scratchSize = denoiserSizes.withoutOverlapScratchSizeInBytes;
    size_t stateSize = denoiserSizes.stateSizeInBytes;
    TOY_CUDA_CHECK_OR_THROW(cudaMalloc(&intensity_d.ptr, sizeof(float)), );
    TOY_CUDA_CHECK_OR_THROW(cudaMalloc(&scratch_d.ptr, scratchSize), );
    TOY_CUDA_CHECK_OR_THROW(cudaMalloc(&state_d.ptr, stateSize), );

    CudaMemoryRAII denoisedFrame_d{};
    TOY_CUDA_CHECK_OR_THROW(
        cudaMalloc(&denoisedFrame_d.ptr,
                   frameWidth * frameHeight * 3 * sizeof(float)), );
    TOY_OPTIX_CHECK_OR_THROW(optixDenoiserSetup(denoiser,
                                                0,
                                                frameWidth,
                                                frameHeight,
                                                (CUdeviceptr)state_d.ptr,
                                                stateSize,
                                                (CUdeviceptr)scratch_d.ptr,
                                                scratchSize), );
    OptixImage2D inputFrame;
    inputFrame.data = (CUdeviceptr)outputFrameBuffer.ptr;
    inputFrame.width = frameWidth;
    inputFrame.height = frameHeight;
    inputFrame.rowStrideInBytes = frameWidth * 3 * sizeof(float);
    inputFrame.pixelStrideInBytes = 3 * sizeof(float);
    inputFrame.format = OPTIX_PIXEL_FORMAT_FLOAT3;

    OptixImage2D denoisedFrame;
    denoisedFrame.data = (CUdeviceptr)denoisedFrame_d.ptr;
    denoisedFrame.width = frameWidth;
    denoisedFrame.height = frameHeight;
    denoisedFrame.rowStrideInBytes = frameWidth * 3 * sizeof(float);
    denoisedFrame.pixelStrideInBytes = 3 * sizeof(float);
    denoisedFrame.format = OPTIX_PIXEL_FORMAT_FLOAT3;

    TOY_OPTIX_CHECK_OR_THROW(
        optixDenoiserComputeIntensity(denoiser,
                                      0,
                                      &inputFrame,
                                      (CUdeviceptr)intensity_d.ptr,
                                      (CUdeviceptr)scratch_d.ptr,
                                      scratchSize), );
    OptixDenoiserParams denoiseParams;
    denoiseParams.denoiseAlpha = 0;
    denoiseParams.hdrIntensity = (CUdeviceptr)intensity_d.ptr;
    denoiseParams.blendFactor = 0.0f;
    TOY_OPTIX_CHECK_OR_THROW(optixDenoiserInvoke(denoiser,
                                                 0,
                                                 &denoiseParams,
                                                 (CUdeviceptr)state_d.ptr,
                                                 stateSize,
                                                 &inputFrame,
                                                 1,
                                                 0,
                                                 0,
                                                 &denoisedFrame,
                                                 (CUdeviceptr)scratch_d.ptr,
                                                 scratchSize), );

    std::swap(denoisedFrame_d, outputFrameBuffer);
    optixDenoiserDestroy(denoiser);
  }
  // write image to output
  {
    auto elemCnt = frameWidth * frameHeight * 3;
    auto result = std::make_unique<float[]>(elemCnt);
    TOY_CUDA_CHECK_OR_THROW(cudaMemcpy((void *)result.get(),
                                       outputFrameBuffer.ptr,
                                       outputBufferSize,
                                       cudaMemcpyDeviceToHost), );
    auto resultAsBytes = std::make_unique<unsigned char[]>(elemCnt);
    auto convertPixel = [](float v) {
      // convert to sRGB
      v = powf(v, 1.0f / 2.2f);
      v *= 255;
      if (v < 0) {
        v = 0;
      }
      if (v > 255) {
        v = 255;
      }
      return (unsigned char)(v);
    };
    auto toneMapping = [&scene](float v) {
      if (!scene.frame.hdr) {
        return v;
      }
      return 1.0f - exp(-scene.frame.exposure * v);
    };
    for (int i = 0; i < elemCnt; i++) {
      resultAsBytes[i] = convertPixel(toneMapping(result[i]));
    }
    stbi_flip_vertically_on_write(1);
    stbi_write_hdr("output.hdr", frameWidth, frameHeight, 3, result.get());
    stbi_write_png(
        "output.png", frameWidth, frameHeight, 3, resultAsBytes.get(), 0);
    printf("result written to output.png\n");
  }
  return 0;
}

int main(int argc, const char **argv) {
  try {
    return run(argc, argv);
  } catch (std::exception &e) {
    std::cerr << e.what() << std::endl;
    return -1;
  }
}