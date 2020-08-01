#include "Context.h"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <optix_function_table_definition.h>
#include <sstream>

// convert path seperator to '/'
void convert_path_seperator(std::string &s) {
  for (auto &c : s) {
    if (c == '\\') {
      c = '/';
    }
  }
}

// return index to seperator
int parent_dir_len(const char *dir) {
  int len = strlen(dir);
  int pLen = 0;
  for (int i = len - 1; i >= 0; i--) {
    if (dir[i] == '/' || dir[i] == '\\') {
      pLen = i;
      break;
    }
  }
  return pLen;
}

std::string parent_dir(const char *dir, bool withSeperator) {
  auto pLen = parent_dir_len(dir);
  if (withSeperator) {
    pLen += 1;
  }
  return std::string(dir, dir + pLen);
}

static const std::string &get_ptx(const std::string &file) {
  static std::map<std::string, std::string> s_PtxCache;

  auto path = Context::ptxDir + "/" + file;
  convert_path_seperator(path);

  {
    auto it = s_PtxCache.find(path);
    if (it != s_PtxCache.end()) {
      return it->second;
    }
  }
  std::ifstream ptxFile(path);
  if (!ptxFile.good()) {
    throw "can not open file " + path;
  }
  auto result = s_PtxCache.emplace(
      std::piecewise_construct,
      std::forward_as_tuple(path),
      std::forward_as_tuple(std::istreambuf_iterator<char>(ptxFile),
                            std::istreambuf_iterator<char>()));
  assert(result.second);
  return result.first->second;
}

OptixDeviceContext Context::context = 0;
std::string Context::ptxDir{};

static void optix_log_callback(unsigned int level,
                               const char *tag,
                               const char *message,
                               void *cbdata) {
  if (level == 0) {
    return;
  }
  if (level >= 4) {
    printf("Optix Info [%s]: %s\n", tag, message);
    return;
  }
  if (level >= 3) {
    printf("Optix Warning [%s]: %s\n", tag, message);
    return;
  }
  fprintf(stderr, "Optix Error [%s]: %s\n", tag, message);
}

void Context::init() {
  TOY_CU_CHECK_OR_THROW(cuInit(0), );
  TOY_CUDA_CHECK_OR_THROW(cudaFree(0), );
  TOY_OPTIX_CHECK_OR_THROW(optixInit(), );
  {
    OptixDeviceContextOptions options{};
    options.logCallbackFunction = optix_log_callback;
    options.logCallbackLevel = 4;
    options.logCallbackData = nullptr;
    TOY_OPTIX_CHECK_OR_THROW(optixDeviceContextCreate(0, &options, &context), );
  }

  std::cout << "Optix initialized. Version " << OPTIX_VERSION << std::endl;
}

static void common_add_group(std::vector<PipelineBuilder::Group> &groups,
                             const std::string &module,
                             const std::string &entry) {
  PipelineBuilder::Group group;
  group.module = module;
  group.entry = entry;
  groups.emplace_back(std::move(group));
}

PipelineBuilder &PipelineBuilder::set_launch_params(const std::string &name) {
  launchParams = name;
  return *this;
}

PipelineBuilder &PipelineBuilder::add_raygen_group(const std::string &module,
                                                   const std::string &entry) {
  common_add_group(raygenGroups, module, entry);
  return *this;
}

PipelineBuilder &
PipelineBuilder::add_exception_group(const std::string &module,
                                     const std::string &entry) {
  common_add_group(exceptionGroups, module, entry);
  return *this;
}

PipelineBuilder &PipelineBuilder::add_miss_group(const std::string &module,
                                                 const std::string &entry) {
  common_add_group(missGroups, module, entry);
  return *this;
}

PipelineBuilder &PipelineBuilder::add_hit_group(const std::string &moduleCH,
                                                const std::string &entryCH,
                                                const std::string &moduleAH,
                                                const std::string &entryAH) {
  HitGroup group;
  group.closestHit.module = moduleCH;
  group.closestHit.entry = entryCH;
  group.anyHit.module = moduleAH;
  group.anyHit.entry = entryAH;
  hitGroups.emplace_back(std::move(group));
  return *this;
}

template <typename T>
void get_unique_modules(std::set<std::string> &modules,
                        const std::vector<T> &groups) {
  for (const auto &g : groups) {
    modules.insert(g.module);
  }
}

Pipeline PipelineBuilder::build() {
  Pipeline pipeline;

  OptixPipelineCompileOptions pipelineCompileOptions{};
  pipelineCompileOptions.usesMotionBlur = 0;
  pipelineCompileOptions.traversableGraphFlags =
      OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
  pipelineCompileOptions.numPayloadValues = 2;
  pipelineCompileOptions.numAttributeValues = 2;
  pipelineCompileOptions.exceptionFlags =
      OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_USER;
  pipelineCompileOptions.pipelineLaunchParamsVariableName =
      launchParams.c_str();

  // compile modules
  std::set<std::string> uniqueModules;
  get_unique_modules(uniqueModules, raygenGroups);
  get_unique_modules(uniqueModules, exceptionGroups);
  get_unique_modules(uniqueModules, missGroups);
  for (const auto &g : hitGroups) {
    uniqueModules.insert(g.closestHit.module);
    uniqueModules.insert(g.anyHit.module);
  }

  for (const auto &moduleName : uniqueModules) {
    OptixModuleCompileOptions moduleOptions{};
    // no explicit limit
    moduleOptions.maxRegisterCount = 0;
    // moduleOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    // moduleOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    moduleOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
    moduleOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    const auto &ptx = get_ptx(moduleName);
    OptixModule module;
    TOY_OPTIX_CHECK_OR_THROW(optixModuleCreateFromPTX(Context::context,
                                                      &moduleOptions,
                                                      &pipelineCompileOptions,
                                                      ptx.c_str(),
                                                      ptx.size(),
                                                      0,
                                                      0,
                                                      &module), );

    pipeline.modules[moduleName] = module;
  }
  // create groups
  {
    {
      auto count = raygenGroups.size();
      pipeline.raygenGroups().resize(count);
      std::vector<OptixProgramGroupOptions> options;
      options.resize(count);
      std::vector<OptixProgramGroupDesc> desc;
      desc.resize(count);
      for (size_t i = 0; i < count; i++) {
        auto &raygenDesc = desc[i];
        const auto &group = raygenGroups[i];
        raygenDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygenDesc.flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
        raygenDesc.raygen.module = pipeline.modules[group.module];
        raygenDesc.raygen.entryFunctionName = group.entry.c_str();
      }
      TOY_OPTIX_CHECK_OR_THROW(
          optixProgramGroupCreate(Context::context,
                                  &desc[0],
                                  count,
                                  &options[0],
                                  0,
                                  0,
                                  &pipeline.raygenGroups()[0]), );
    }
    {
      auto count = missGroups.size();
      pipeline.missGroups().resize(count);
      std::vector<OptixProgramGroupOptions> options;
      options.resize(count);
      std::vector<OptixProgramGroupDesc> desc;
      desc.resize(count);
      for (size_t i = 0; i < count; i++) {
        auto &missDesc = desc[i];
        const auto &group = missGroups[i];
        missDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        missDesc.flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
        missDesc.miss.module = pipeline.modules[group.module];
        missDesc.miss.entryFunctionName = group.entry.c_str();
      }
      TOY_OPTIX_CHECK_OR_THROW(
          optixProgramGroupCreate(Context::context,
                                  &desc[0],
                                  count,
                                  &options[0],
                                  0,
                                  0,
                                  &pipeline.missGroups()[0]), );
    }
    {
      auto count = exceptionGroups.size();
      pipeline.exceptionGroups().resize(count);
      std::vector<OptixProgramGroupOptions> options;
      options.resize(count);
      std::vector<OptixProgramGroupDesc> desc;
      desc.resize(count);
      for (size_t i = 0; i < count; i++) {
        auto &exceptionDesc = desc[i];
        const auto &group = exceptionGroups[i];
        exceptionDesc.kind = OPTIX_PROGRAM_GROUP_KIND_EXCEPTION;
        exceptionDesc.flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
        exceptionDesc.exception.module = pipeline.modules[group.module];
        exceptionDesc.exception.entryFunctionName = group.entry.c_str();
      }
      TOY_OPTIX_CHECK_OR_THROW(
          optixProgramGroupCreate(Context::context,
                                  &desc[0],
                                  count,
                                  &options[0],
                                  0,
                                  0,
                                  &pipeline.exceptionGroups()[0]), );
    }
    {
      auto count = hitGroups.size();
      pipeline.hitGroups().resize(count);
      std::vector<OptixProgramGroupOptions> options;
      options.resize(count);
      std::vector<OptixProgramGroupDesc> desc;
      desc.resize(count);
      for (size_t i = 0; i < count; i++) {
        auto &hitDesc = desc[i];
        const auto &group = hitGroups[i];

        hitDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hitDesc.flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
        hitDesc.hitgroup.moduleCH = pipeline.modules[group.closestHit.module];
        hitDesc.hitgroup.entryFunctionNameCH = group.closestHit.entry.c_str();
        hitDesc.hitgroup.moduleAH = pipeline.modules[group.anyHit.module];
        hitDesc.hitgroup.entryFunctionNameAH = group.anyHit.entry.c_str();
        hitDesc.hitgroup.moduleIS = 0;
        hitDesc.hitgroup.entryFunctionNameIS = 0;
      }
      TOY_OPTIX_CHECK_OR_THROW(
          optixProgramGroupCreate(Context::context,
                                  &desc[0],
                                  count,
                                  &options[0],
                                  0,
                                  0,
                                  &pipeline.hitGroups()[0]), );
    }
    // link pipeline
    {
      OptixPipelineLinkOptions linkOptions;
      linkOptions.maxTraceDepth = 2;
      linkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
      std::vector<OptixProgramGroup> groups;
      groups.reserve(raygenGroups.size() + missGroups.size() +
                     exceptionGroups.size() + hitGroups.size());
      for (int i = 0; i < Pipeline::groupCnt; i++) {
        for (auto g : pipeline.groups[i]) {
          groups.push_back(g);
        }
      }
      TOY_OPTIX_CHECK_OR_THROW(optixPipelineCreate(Context::context,
                                                   &pipelineCompileOptions,
                                                   &linkOptions,
                                                   &groups[0],
                                                   groups.size(),
                                                   0,
                                                   0,
                                                   &pipeline.pipeline), );
    }
  }
  return pipeline;
}

ShaderBindingTable ShaderBindingTableBuilder::build() {
  ShaderBindingTable sbt;
  for (int i = 0; i < Pipeline::groupCnt; i++) {
    // calculate stride and size
    size_t stride = 0;
    for (const auto &record : records[i]) {
      stride = std::max(stride, record.size);
    }
    size_t size = records[i].size() * stride;
    if (!sbt.sbtBuffers_d[i] || sbt.sbtBufferSizes[i] < size) {
      TOY_CUDA_CHECK_OR_THROW(cudaFree(sbt.sbtBuffers_d[i]), );
      TOY_CUDA_CHECK_OR_THROW(cudaMalloc(&sbt.sbtBuffers_d[i], size), );
    }
    sbt.sbtBufferSizes[i] = size;
    sbt.sbtBufferStrides[i] = stride;
    // update data
    for (size_t ri = 0; ri < records[i].size(); ri++) {
      const auto &record = records[i][ri];
      void *buf_d = (void *)((char *)sbt.sbtBuffers_d[i] + ri * stride);
      TOY_CUDA_CHECK_OR_THROW(
          cudaMemcpy(
              buf_d, record.record, record.size, cudaMemcpyHostToDevice), );
    }
  }

  // udpate sbt
  sbt.sbt.raygenRecord =
      (CUdeviceptr)sbt.sbtBuffers_d[Pipeline::raygenGroupIndex];
  sbt.sbt.exceptionRecord =
      (CUdeviceptr)sbt.sbtBuffers_d[Pipeline::exceptionGroupIndex];
  sbt.sbt.missRecordBase =
      (CUdeviceptr)sbt.sbtBuffers_d[Pipeline::missGroupIndex];
  sbt.sbt.missRecordStrideInBytes =
      sbt.sbtBufferStrides[Pipeline::missGroupIndex];
  sbt.sbt.missRecordCount = records[Pipeline::missGroupIndex].size();
  sbt.sbt.hitgroupRecordBase =
      (CUdeviceptr)sbt.sbtBuffers_d[Pipeline::hitGroupIndex];
  sbt.sbt.hitgroupRecordStrideInBytes =
      sbt.sbtBufferStrides[Pipeline::hitGroupIndex];
  sbt.sbt.hitgroupRecordCount = records[Pipeline::hitGroupIndex].size();
  sbt.sbt.callablesRecordBase = 0;
  sbt.sbt.callablesRecordStrideInBytes = 0;
  sbt.sbt.callablesRecordCount = 0;
  return sbt;
}

ShaderBindingTableBuilder::~ShaderBindingTableBuilder() {
  for (auto &record : records) {
    for (auto &r : record) {
      r.destroy();
    }
  }
}

inline std::string pretty_format_bytes(size_t size) {
  std::string postfix = "Bytes";
  double sd = size;
  if (sd > 1024) {
    sd /= 1024.0;
    postfix = "KB";
  }
  if (sd > 1024) {
    sd /= 1024.0;
    postfix = "MB";
  }
  if (sd > 1024.0) {
    sd /= 1024.0;
    postfix = "GB";
  }
  std::stringstream ss;
  ss << std::fixed << std::setprecision(2) << sd << postfix;
  return ss.str();
}

GAS GASBuilder::build() {
  GAS gas{};
  assert(vertices);
  assert(indices);
  assert(materialIds);
  // alloc buffers for vertices and indices
  size_t vertexBufferSize = vertexCount * sizeof(float) * 3;
  size_t indexBufferSize = primitiveCount * sizeof(unsigned int) * 3;
  size_t materialIdBufferSize = primitiveCount * sizeof(unsigned int);
  TOY_CUDA_CHECK_OR_THROW(cudaMalloc(&gas.vertices_d, vertexBufferSize), );
  TOY_CUDA_CHECK_OR_THROW(cudaMalloc(&gas.indices_d, indexBufferSize), );
  TOY_CUDA_CHECK_OR_THROW(
      cudaMalloc(&gas.materialIds_d, materialIdBufferSize), );
  // copy data to cuda
  TOY_CUDA_CHECK_OR_THROW(cudaMemcpy(gas.vertices_d,
                                     vertices,
                                     vertexBufferSize,
                                     cudaMemcpyHostToDevice), );
  TOY_CUDA_CHECK_OR_THROW(
      cudaMemcpy(
          gas.indices_d, indices, indexBufferSize, cudaMemcpyHostToDevice), );
  TOY_CUDA_CHECK_OR_THROW(cudaMemcpy(gas.materialIds_d,
                                     materialIds,
                                     materialIdBufferSize,
                                     cudaMemcpyHostToDevice), );
  // build gas
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
  triangles.vertexBuffers = (CUdeviceptr *)(&gas.vertices_d);
  triangles.numVertices = vertexCount;
  triangles.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
  triangles.vertexStrideInBytes = sizeof(float3);
  triangles.indexBuffer = (CUdeviceptr)gas.indices_d;
  triangles.numIndexTriplets = primitiveCount;
  triangles.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
  triangles.indexStrideInBytes = sizeof(uint3);
  triangles.preTransform = 0;
  auto flags = std::make_unique<unsigned int[]>(materialCount);
  for (int i = 0; i < materialCount; i++) {
    flags[i] = OPTIX_GEOMETRY_FLAG_NONE;
  }
  triangles.flags = flags.get();
  triangles.numSbtRecords = materialCount;
  triangles.sbtIndexOffsetBuffer = (CUdeviceptr)(gas.materialIds_d);
  triangles.sbtIndexOffsetSizeInBytes = sizeof(unsigned int);
  triangles.sbtIndexOffsetStrideInBytes = sizeof(unsigned int);
  triangles.primitiveIndexOffset = 0;

  // alloc buffers
  OptixAccelBufferSizes bufferSizes{};
  TOY_OPTIX_CHECK_OR_THROW(
      optixAccelComputeMemoryUsage(
          Context::context, &options, &buildInput, 1, &bufferSizes), );
  TOY_CUDA_CHECK_OR_THROW(
      cudaMalloc(&gas.accel_d, bufferSizes.outputSizeInBytes), );
  CudaMemoryRAII tmpBuffer{};
  TOY_CUDA_CHECK_OR_THROW(
      cudaMalloc(&tmpBuffer.ptr, bufferSizes.tempSizeInBytes), );
  // from optix SDK samples, and comments of cudaMalloc, it seems safe to assume
  // memory returned from cudaMalloc is properly aligned
  assert((size_t)(gas.accel_d) % OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT == 0);
  assert((size_t)(tmpBuffer.ptr) % OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT == 0);

  CudaMemoryRAII compactedSize_d;
  TOY_CUDA_CHECK_OR_THROW(cudaMalloc(&compactedSize_d.ptr, sizeof(size_t)), );
  OptixAccelEmitDesc property;
  property.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
  property.result = (CUdeviceptr)compactedSize_d.ptr;
  TOY_OPTIX_CHECK_OR_THROW(optixAccelBuild(Context::context,
                                           0,
                                           &options,
                                           &buildInput,
                                           1,
                                           (CUdeviceptr)tmpBuffer.ptr,
                                           bufferSizes.tempSizeInBytes,
                                           (CUdeviceptr)gas.accel_d,
                                           bufferSizes.outputSizeInBytes,
                                           &gas.gas,
                                           &property,
                                           1), );
  TOY_CUDA_CHECK_OR_THROW(cudaStreamSynchronize(0), );
  // do compaction
  size_t compactedSize;
  TOY_CUDA_CHECK_OR_THROW(cudaMemcpy(&compactedSize,
                                     compactedSize_d.ptr,
                                     sizeof(size_t),
                                     cudaMemcpyDeviceToHost), );
  if (compactedSize < bufferSizes.outputSizeInBytes) {
    std::cout << "Compact acceleration structure. Uncompacted size = "
              << pretty_format_bytes(bufferSizes.outputSizeInBytes)
              << ", compacted size = " << pretty_format_bytes(compactedSize)
              << std::endl;
    void *compactedAccel_d;
    TOY_CUDA_CHECK_OR_THROW(cudaMalloc(&compactedAccel_d, compactedSize), );
    OptixTraversableHandle compactedAccelHandle;
    TOY_OPTIX_CHECK_OR_THROW(optixAccelCompact(Context::context,
                                               0,
                                               gas.gas,
                                               (CUdeviceptr)compactedAccel_d,
                                               compactedSize,
                                               &compactedAccelHandle), );
    TOY_CUDA_CHECK_OR_THROW(cudaStreamSynchronize(0), );
    // free uncompacted accel
    TOY_CUDA_CHECK_OR_THROW(cudaFree(gas.accel_d), );
    gas.accel_d = compactedAccel_d;
    gas.gas = compactedAccelHandle;
  }

  return gas;
}

void Pipeline::destroy() {
  for (auto &pairs : modules) {
    optixModuleDestroy(pairs.second);
  }
  optixPipelineDestroy(pipeline);
}

void ShaderBindingTable::destroy() {
  for (auto buf : sbtBuffers_d) {
    cudaFree(buf);
  }
}

void GAS::destroy() {
  cudaFree(accel_d);
  cudaFree(vertices_d);
  cudaFree(indices_d);
  cudaFree(materialIds_d);
}