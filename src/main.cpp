#include "Context.h"
#include "exceptions.h"
#include "pipeline.h"
#include "vec_math.h"
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

template <typename T> struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) Record {
  char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  T data;
};

struct OptixState {
  OptixDeviceContext context = 0;
  OptixTraversableHandle gas = 0;
  OptixModule module = 0;

  constexpr static int rayGenGroupIndex = 0;
  constexpr static int missGroupIndex = 1;
  constexpr static int exceptionGroupIndex = 2;
  constexpr static int hitGroupIndex = 3;
  constexpr static int groupCnt = 4;

  OptixProgramGroup groups[groupCnt]{0};
  OptixPipeline pipeline = 0;

  OptixShaderBindingTable sbt;
  CudaMemoryRAII raygenRecord;
  CudaMemoryRAII exceptionRecord;
  CudaMemoryRAII missRecords;
  CudaMemoryRAII hitRecords;

  CudaMemoryRAII launchParams;

  CudaMemoryRAII outputFrameBuffer;

  OptixState() = default;
  OptixState(const OptixState &) = delete;
  OptixState(OptixState &&) = delete;
  OptixState &operator=(const OptixState &) = delete;
  OptixState &operator=(OptixState &&) = delete;

  ~OptixState() {
    optixPipelineDestroy(pipeline);
    for (int i = 0; i < groupCnt; i++) {
      optixProgramGroupDestroy(groups[i]);
    }
    optixModuleDestroy(module);
    optixDeviceContextDestroy(context);
  }
};

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
  // copy data to device
  TOY_CUDA_CHECK_OR_THROW(cudaFree(0), );
  static_assert(std::is_same_v<tinyobj::real_t, float>,
                "only float data is supported");
  CudaMemoryRAII vertices_d{};
  int numVertices = attrib.vertices.size();
  {
    auto verticesBufferSize = numVertices * sizeof(float);
    TOY_CUDA_CHECK_OR_THROW(cudaMalloc(&vertices_d.ptr, verticesBufferSize), );
    TOY_CUDA_CHECK_OR_THROW(cudaMemcpy(vertices_d.ptr,
                                       attrib.vertices.data(),
                                       verticesBufferSize,
                                       cudaMemcpyHostToDevice), );
  }
  // per vertices indices
  CudaMemoryRAII indices_d{};
  // should be multiple of 3
  int indicesCnt = 0;
  {
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
    auto bufSize = sizeof(unsigned int) * indicesCnt;
    TOY_CUDA_CHECK_OR_THROW(cudaMalloc(&indices_d.ptr, bufSize), );
    TOY_CUDA_CHECK_OR_THROW(
        cudaMemcpy(
            indices_d.ptr, indices_h.get(), bufSize, cudaMemcpyHostToDevice), );
  }
  // transform material data to target format
  // 0 if use default material
  int materialCnt = materials.size();
  std::vector<Record<HitGroupData>> hitRecords(materialCnt == 0 ? 1
                                                                : materialCnt);
  if (materialCnt == 0) {
    // default material
    hitRecords[0].data.baseColor = make_float3(0.5f, 0.5f, 0.5f);
    hitRecords[0].data.emission = make_float3(0, 0, 0);
  } else {
    for (int i = 0; i < materialCnt; i++) {
      auto &data = hitRecords[i].data;
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

  // per premitive material indices
  CudaMemoryRAII materialIds_d{};
  int primitiveCnt = indicesCnt / 3;
  if (materialCnt > 1) {
    auto buf = std::make_unique<unsigned int[]>(primitiveCnt);
    int faceCnt = 0;
    for (const auto &shape : shapes) {
      const auto &mats = shape.mesh.material_ids;
      memcpy(buf.get() + faceCnt, mats.data(), mats.size() * sizeof(int));
      faceCnt += mats.size();
    }
    assert(faceCnt == primitiveCnt);
    auto bufSize = primitiveCnt * sizeof(unsigned int);
    TOY_CUDA_CHECK_OR_THROW(cudaMalloc(&materialIds_d.ptr, bufSize), );
    TOY_CUDA_CHECK_OR_THROW(
        cudaMemcpy(
            materialIds_d.ptr, buf.get(), bufSize, cudaMemcpyHostToDevice), );
  }

  // init optix
  OptixState state;
  Context::init();
  state.context = Context::context;
  // build gas
  CudaMemoryRAII accelOutputBuffer{};
  {
    OptixAccelBuildOptions options{};
    options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION |
                         OPTIX_BUILD_FLAG_PREFER_FAST_TRACE |
                         OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    // no motion
    options.motionOptions.numKeys = 1;
    options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixBuildInput buildInput{};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    auto &triangles = buildInput.triangleArray;
    triangles.vertexBuffers = (CUdeviceptr *)(&vertices_d.ptr);
    triangles.numVertices = numVertices;
    triangles.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangles.vertexStrideInBytes = sizeof(float) * 3;
    triangles.indexBuffer = (CUdeviceptr)indices_d.ptr;
    triangles.numIndexTriplets = primitiveCnt;
    triangles.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangles.indexStrideInBytes = sizeof(unsigned int) * 3;
    triangles.preTransform = 0;
    auto sbtNum = materialCnt > 0 ? materialCnt : 1;
    auto flags = std::make_unique<unsigned int[]>(sbtNum);
    for (int i = 0; i < sbtNum; i++) {
      flags[i] = OPTIX_GEOMETRY_FLAG_NONE;
    }
    if (materialCnt > 1) {
      triangles.flags = flags.get();
      triangles.numSbtRecords = materialCnt;
      triangles.sbtIndexOffsetBuffer = (CUdeviceptr)materialIds_d.ptr;
      triangles.sbtIndexOffsetSizeInBytes = sizeof(unsigned int);
      triangles.sbtIndexOffsetStrideInBytes = sizeof(unsigned int);
    } else {
      triangles.flags = flags.get();
      triangles.numSbtRecords = 1;
      triangles.sbtIndexOffsetBuffer = 0;
      triangles.sbtIndexOffsetSizeInBytes = 0;
      triangles.sbtIndexOffsetStrideInBytes = 0;
    }
    triangles.primitiveIndexOffset = 0;

    // alloc buffers
    OptixAccelBufferSizes bufferSizes{};
    TOY_OPTIX_CHECK_OR_THROW(
        optixAccelComputeMemoryUsage(
            state.context, &options, &buildInput, 1, &bufferSizes), );
    TOY_CUDA_CHECK_OR_THROW(
        cudaMalloc(&accelOutputBuffer.ptr, bufferSizes.outputSizeInBytes), );
    CudaMemoryRAII tmpBuffer{};
    TOY_CUDA_CHECK_OR_THROW(
        cudaMalloc(&tmpBuffer.ptr, bufferSizes.tempSizeInBytes), );
    // from optix SDK samples, it seems safe to assume memory returned from
    // cudaMalloc is properly aligned
    assert((size_t)(accelOutputBuffer.ptr) %
               OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT ==
           0);
    assert((size_t)(tmpBuffer.ptr) % OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT == 0);
    TOY_OPTIX_CHECK_OR_THROW(optixAccelBuild(state.context,
                                             0,
                                             &options,
                                             &buildInput,
                                             1,
                                             (CUdeviceptr)tmpBuffer.ptr,
                                             bufferSizes.tempSizeInBytes,
                                             (CUdeviceptr)accelOutputBuffer.ptr,
                                             bufferSizes.outputSizeInBytes,
                                             &state.gas,
                                             nullptr,
                                             0), );
    TOY_CUDA_CHECK_OR_THROW(cudaStreamSynchronize(0), );
  }
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
  state.pipeline = pipeline.pipeline;
  state.module = pipeline.modules[moduleName];
  state.groups[state.rayGenGroupIndex] = pipeline.raygenGroups[0];
  state.groups[state.missGroupIndex] = pipeline.missGroups[0];
  state.groups[state.exceptionGroupIndex] = pipeline.exceptionGroups[0];
  state.groups[state.hitGroupIndex] = pipeline.hitGroups[0];
  // set up sbt
  {
    {
      Record<RayGenData> raygenRecord;
      TOY_OPTIX_CHECK_OR_THROW(
          optixSbtRecordPackHeader(state.groups[state.rayGenGroupIndex],
                                   &raygenRecord), );
      TOY_CUDA_CHECK_OR_THROW(
          cudaMalloc(&state.raygenRecord.ptr, sizeof(raygenRecord)), );
      TOY_CUDA_CHECK_OR_THROW(cudaMemcpy(state.raygenRecord.ptr,
                                         &raygenRecord,
                                         sizeof(raygenRecord),
                                         cudaMemcpyHostToDevice), );
      state.sbt.raygenRecord = (CUdeviceptr)state.raygenRecord.ptr;
    }
    {
      Record<ExceptionData> exceptionRecord;
      exceptionRecord.data.errorColor = make_float3(0, 1.0f, 0);
      TOY_OPTIX_CHECK_OR_THROW(
          optixSbtRecordPackHeader(state.groups[state.exceptionGroupIndex],
                                   &exceptionRecord), );
      TOY_CUDA_CHECK_OR_THROW(
          cudaMalloc(&state.exceptionRecord.ptr, sizeof(exceptionRecord)), );
      TOY_CUDA_CHECK_OR_THROW(cudaMemcpy(state.exceptionRecord.ptr,
                                         &exceptionRecord,
                                         sizeof(exceptionRecord),
                                         cudaMemcpyHostToDevice), );
      state.sbt.exceptionRecord = (CUdeviceptr)state.exceptionRecord.ptr;
    }
    {
      Record<MissData> missRecord;
      TOY_OPTIX_CHECK_OR_THROW(
          optixSbtRecordPackHeader(state.groups[state.missGroupIndex],
                                   &missRecord), );
      missRecord.data.color = make_float3(0.0f, 0.0f, 0.0f);
      TOY_CUDA_CHECK_OR_THROW(
          cudaMalloc(&state.missRecords.ptr, sizeof(missRecord)), );
      TOY_CUDA_CHECK_OR_THROW(cudaMemcpy(state.missRecords.ptr,
                                         &missRecord,
                                         sizeof(missRecord),
                                         cudaMemcpyHostToDevice), );
      state.sbt.missRecordBase = (CUdeviceptr)state.missRecords.ptr;
      state.sbt.missRecordStrideInBytes = sizeof(missRecord);
      state.sbt.missRecordCount = 1;
    }
    {
      for (auto &record : hitRecords) {
        TOY_OPTIX_CHECK_OR_THROW(
            optixSbtRecordPackHeader(state.groups[state.hitGroupIndex],
                                     &record), );
      }
      auto hitRecordCnt = hitRecords.size();
      auto hitRecordSize = sizeof(Record<HitGroupData>);
      auto hitRecordBufSize = hitRecordCnt * hitRecordSize;
      TOY_CUDA_CHECK_OR_THROW(
          cudaMalloc(&state.hitRecords.ptr, hitRecordBufSize), );
      TOY_CUDA_CHECK_OR_THROW(cudaMemcpy(state.hitRecords.ptr,
                                         hitRecords.data(),
                                         hitRecordBufSize,
                                         cudaMemcpyHostToDevice), );
      state.sbt.hitgroupRecordBase = (CUdeviceptr)state.hitRecords.ptr;
      state.sbt.hitgroupRecordStrideInBytes = hitRecordSize;
      state.sbt.hitgroupRecordCount = hitRecordCnt;
    }
    {
      state.sbt.callablesRecordBase = 0;
      state.sbt.callablesRecordStrideInBytes = 0;
      state.sbt.callablesRecordCount = 0;
    }
  }
  unsigned int frameWidth = 800, frameHeight = 800;
  unsigned int tileWidth = 64, tileHeight = 64;
  float frameResolution = 1000;
  auto outputBufferSize = frameWidth * frameHeight * sizeof(float3);
  constexpr unsigned int numStreams = 8;

  cudaStream_t streams[numStreams]{};
  for (int i = 0; i < numStreams; i++) {
    TOY_CUDA_CHECK_OR_THROW(cudaStreamCreate(&streams[i]), );
  }

  // launch
  {
    // alloc output frame buffer
    TOY_CUDA_CHECK_OR_THROW(
        cudaMalloc(&state.outputFrameBuffer.ptr, outputBufferSize), );

    LaunchParams launchParams[numStreams]{};
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
      camera.canvas = make_rect(
          -canvasWidth / 2.0f, -canvasHeight / 2.0f, canvasWidth, canvasHeight);

      auto &frame = param.outputFrame;
      frame.width = frameWidth;
      frame.height = frameHeight;
      frame.buffer = (float3 *)state.outputFrameBuffer.ptr;

      auto &scene = param.scene;
      scene.gas = state.gas;
      scene.epsilon = 1e-5;
      scene.extent = 10.0f;

      param.spp = 100;
      param.maxPathLength = 15;
    }

    auto startTime = std::chrono::system_clock::now();

    {
      auto t = std::chrono::system_clock::to_time_t(startTime);
      std::cout << "render start at " << std::ctime(&t) << std::flush;
    }
    {
      TOY_CUDA_CHECK_OR_THROW(cudaMalloc(&state.launchParams.ptr,
                                         sizeof(LaunchParams) * numStreams), );
      int streamIndex = 0;
      // tile frame for launch
      for (int x = 0; x < frameWidth; x += tileWidth) {
        for (int y = 0; y < frameHeight;
             y += tileHeight, streamIndex = (streamIndex + 1) % numStreams) {
          auto stream = streams[streamIndex];
          auto &param = launchParams[streamIndex];
          auto paramPtrD = (void *)((char *)state.launchParams.ptr +
                                    streamIndex * sizeof(LaunchParams));
          // init tile
          int launchWidth = min(tileWidth, frameWidth - x);
          int launchHeight = min(tileHeight, frameHeight - y);
          auto &frame = param.outputFrame;
          frame.tile = make_rect_int(x, y, launchWidth, launchHeight);

          TOY_CUDA_CHECK_OR_THROW(cudaMemcpyAsync(paramPtrD,
                                                  &param,
                                                  sizeof(LaunchParams),
                                                  cudaMemcpyHostToDevice,
                                                  stream), );

          // launch size should match exactly with cropped size of the ouput
          // frame
          TOY_OPTIX_CHECK_OR_THROW(optixLaunch(state.pipeline,
                                               stream,
                                               (CUdeviceptr)paramPtrD,
                                               sizeof(LaunchParams),
                                               &state.sbt,
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
                                       state.outputFrameBuffer.ptr,
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
