#include "Scene.h"
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
    for (int i = 0; i < elemCnt; i++) {
      resultAsBytes[i] = convertPixel(result[i]);
    }
    stbi_flip_vertically_on_write(1);
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