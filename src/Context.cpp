#include "Context.h"
#include <cassert>
#include <cstdio>
#include <fstream>
#include <optix_function_table_definition.h>

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
  TOY_OPTIX_CHECK_OR_THROW(optixInit(), );
  {
    OptixDeviceContextOptions options{};
    options.logCallbackFunction = optix_log_callback;
    options.logCallbackLevel = 4;
    options.logCallbackData = nullptr;
    TOY_OPTIX_CHECK_OR_THROW(optixDeviceContextCreate(0, &options, &context), );
  }
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
      pipeline.raygenGroups.resize(count);
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
                                  &pipeline.raygenGroups[0]), );
    }
    {
      auto count = missGroups.size();
      pipeline.missGroups.resize(count);
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
                                  &pipeline.missGroups[0]), );
    }
    {
      auto count = exceptionGroups.size();
      pipeline.exceptionGroups.resize(count);
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
                                  &pipeline.exceptionGroups[0]), );
    }
    {
      auto count = hitGroups.size();
      pipeline.hitGroups.resize(count);
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
                                  &pipeline.hitGroups[0]), );
    }
    // link pipeline
    {
      OptixPipelineLinkOptions linkOptions;
      linkOptions.maxTraceDepth = 2;
      linkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
      linkOptions.overrideUsesMotionBlur = 0;
      std::vector<OptixProgramGroup> groups;
      groups.reserve(raygenGroups.size() + missGroups.size() +
                     exceptionGroups.size() + hitGroups.size());
      for (auto g : pipeline.raygenGroups) {
        groups.push_back(g);
      }
      for (auto g : pipeline.missGroups) {
        groups.push_back(g);
      }
      for (auto g : pipeline.exceptionGroups) {
        groups.push_back(g);
      }
      for (auto g : pipeline.hitGroups) {
        groups.push_back(g);
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