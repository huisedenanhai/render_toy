#include "Context.h"
#include "exceptions.h"
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

int main(int argc, const char **argv) {
  printf("exe: %s\n", argv[0]);
  configure_ptx_dir(argv[0]);
  if (argc < 2) {
    printf("no input file specified, stop.");
    return 0;
  }
  // init optix
  Context::init();

  // load obj
  tinyobj::attrib_t attrib;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;
  {
    const char *inputFile = argv[1];
    std::cout << "start load obj file: " << inputFile << std::endl;
    auto materialBaseDir = parent_dir(inputFile, true);
    std::string loadError;
    if (!tinyobj::LoadObj(&attrib,
                          &shapes,
                          &materials,
                          &loadError,
                          inputFile,
                          materialBaseDir.c_str(),
                          true)) {
      std::cerr << "failed to load obj file: " << loadError << std::endl;
      return -1;
    } else {
      std::cout << "finish load obj file " << inputFile;
      if (!loadError.empty()) {
        std::cout << " with info:\n" << loadError;
      }
      std::cout << std::endl;
    }
  }
  static_assert(std::is_same_v<tinyobj::real_t, float>,
                "only float data is supported");

  GASBuilder gasBuilder;
  gasBuilder.vertices = &attrib.vertices[0];
  gasBuilder.vertexCount = attrib.vertices.size() / 3;

  // should be multiple of 3
  int indicesCnt = 0;
  auto shapeCnt = shapes.size();
  for (const auto &shape : shapes) {
    indicesCnt += shape.mesh.indices.size();
  }
  assert(indicesCnt % 3 == 0);
  auto indices_h = std::make_unique<unsigned int[]>(indicesCnt);
  int index = 0;
  for (const auto &shape : shapes) {
    for (const auto &indice : shape.mesh.indices) {
      indices_h[index++] = indice.vertex_index;
    }
  }

  gasBuilder.indices = indices_h.get();
  gasBuilder.primitiveCount = indicesCnt / 3;

  // transform material data to target format
  // 0 if use default material
  int materialCnt = materials.size();
  std::vector<dev::HitGroupData> hitRecords(materialCnt == 0 ? 1 : materialCnt);
  if (materialCnt == 0) {
    // default material
    hitRecords[0].baseColor = make_float3(0.5f, 0.5f, 0.5f);
    hitRecords[0].emission = make_float3(0, 0, 0);
  } else {
    for (int i = 0; i < materialCnt; i++) {
      auto &data = hitRecords[i];
      const auto &mat = materials[i];
      data.baseColor =
          make_float3(mat.diffuse[0], mat.diffuse[1], mat.diffuse[2]);
      data.emission =
          make_float3(mat.emission[0], mat.emission[1], mat.emission[2]) * Pi;
      printf("mat %s {\n", mat.name.c_str());
      printf("  color = (%f, %f, %f)\n",
             data.baseColor.x,
             data.baseColor.y,
             data.baseColor.z);
      printf("  emission = (%f, %f, %f)\n",
             data.emission.x,
             data.emission.y,
             data.emission.z);
      printf("}\n");
    }
  }

  // per primitive material indices
  auto matIds = std::make_unique<unsigned int[]>(gasBuilder.primitiveCount);
  int faceCnt = 0;
  if (materialCnt > 1) {
    for (const auto &shape : shapes) {
      const auto &mats = shape.mesh.material_ids;
      memcpy(matIds.get() + faceCnt, mats.data(), mats.size() * sizeof(int));
      faceCnt += mats.size();
    }
    assert(faceCnt == gasBuilder.primitiveCount);
  } else {
    memset(matIds.get(), 0, gasBuilder.primitiveCount * sizeof(unsigned int));
  }
  gasBuilder.materialIds = matIds.get();
  gasBuilder.materialCount = materialCnt == 0 ? 1 : materialCnt;

  // build accel
  auto gas = gasBuilder.build();
  // assemble pipeline
  auto moduleName = "pipeline.ptx";
  auto pipeline =
      PipelineBuilder()
          .set_launch_params("g_LaunchParams")
          .add_raygen_group(moduleName, "__raygen__entry")
          .add_miss_group(moduleName, "__miss__entry")
          .add_exception_group(moduleName, "__exception__entry")
          .add_hit_group(
              moduleName, "__closesthit__entry", moduleName, "__anyhit__entry")
          .build();
  // set up sbt
  ShaderBindingTable sbt(&pipeline);
  {
    sbt.add_raygen_record<dev::RayGenData>(0);
    auto exceptionRecord = sbt.add_exception_record<dev::ExceptionData>(0);
    exceptionRecord->errorColor = make_float3(0, 1.0f, 0);
    auto missRecord = sbt.add_miss_record<dev::MissData>(0);
    missRecord->color = make_float3(0.0f, 0.0f, 0.0f);
    for (auto &record : hitRecords) {
      auto hitRecord = sbt.add_hit_record<dev::HitGroupData>(0);
      *hitRecord = record;
    }
    sbt.commit();
  }

  unsigned int frameWidth = 800, frameHeight = 800;
  unsigned int tileWidth = 64, tileHeight = 64;
  float frameResolution = 1000;
  unsigned int spp = 100000;
  unsigned int maxPathLength = 15;
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
      camera.position = make_float3(0, 1.0f, 5.0f);
      camera.right = make_float3(1.0f, 0.0f, 0.0f);
      camera.up = make_float3(0.0f, 1.0f, 0.0f);
      camera.back = make_float3(0.0f, 0.0f, 1.0f);
      auto canvasWidth = frameWidth / frameResolution;
      auto canvasHeight = frameHeight / frameResolution;
      camera.canvas = dev::make_rect(
          -canvasWidth / 2.0f, -canvasHeight / 2.0f, canvasWidth, canvasHeight);

      auto &frame = param.outputFrame;
      frame.width = frameWidth;
      frame.height = frameHeight;
      frame.buffer = (float3 *)outputFrameBuffer.ptr;

      auto &scene = param.scene;
      scene.gas = gas.gas;
      scene.epsilon = 1e-5;
      scene.extent = 10.0f;

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
          TOY_OPTIX_CHECK_OR_THROW(optixLaunch(pipeline.pipeline,
                                               stream,
                                               (CUdeviceptr)paramPtrD,
                                               sizeof(dev::LaunchParams),
                                               &sbt.sbt,
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
      if (v < 0) {
        v = 0;
      }
      if (v > 255) {
        v = 255;
      }
      return (unsigned char)(v);
    };
    for (int i = 0; i < elemCnt; i++) {
      resultAsBytes[i] = convertPixel(255 * result[i]);
    }
    stbi_flip_vertically_on_write(1);
    stbi_write_png(
        "output.png", frameWidth, frameHeight, 3, resultAsBytes.get(), 0);
    printf("result written to output.png\n");
  }
  return 0;
}
