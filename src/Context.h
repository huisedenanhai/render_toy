#pragma once

#include "exceptions.h"
#include <iostream>
#include <map>
#include <optix.h>
#include <optix_function_table.h>
#include <optix_stubs.h>
#include <set>
#include <vector>

void convert_path_seperator(std::string &s);
int parent_dir_len(const char *dir);
std::string parent_dir(const char *dir, bool withSeperator = false);

struct CudaMemoryRAII {
  void *ptr = nullptr;

  void release() noexcept {
    try {
      TOY_CUDA_CHECK_OR_THROW(cudaFree(ptr), );
    } catch (const CudaException &e) {
      std::cerr << "failed to release cuda memory: " << e.what() << std::endl;
    }
    ptr = nullptr;
  }

  CudaMemoryRAII(void *p = nullptr) : ptr(p) {}
  CudaMemoryRAII(const CudaMemoryRAII &) = delete;
  CudaMemoryRAII(CudaMemoryRAII &&from) noexcept
      : ptr(std::exchange(from.ptr, nullptr)) {}

  CudaMemoryRAII &operator=(const CudaMemoryRAII &) = delete;
  CudaMemoryRAII &operator=(CudaMemoryRAII &&from) {
    if (ptr == from.ptr) {
      return *this;
    }
    release();
    ptr = std::exchange(from.ptr, nullptr);
    return *this;
  }

  ~CudaMemoryRAII() {
    release();
  }
};

struct Pipeline {
  std::vector<OptixProgramGroup> raygenGroups;
  std::vector<OptixProgramGroup> missGroups;
  std::vector<OptixProgramGroup> exceptionGroups;
  std::vector<OptixProgramGroup> hitGroups;
  std::map<std::string, OptixModule> modules;
  OptixPipeline pipeline;
};

// order of adding will be preserved
struct PipelineBuilder {
  PipelineBuilder &set_launch_params(const std::string &name);
  PipelineBuilder &add_raygen_group(const std::string &module,
                                    const std::string &entry);
  PipelineBuilder &add_exception_group(const std::string &module,
                                       const std::string &entry);
  PipelineBuilder &add_miss_group(const std::string &module,
                                  const std::string &entry);
  PipelineBuilder &add_hit_group(const std::string &moduleCH,
                                 const std::string &entryCH,
                                 const std::string &moduleAH,
                                 const std::string &entryAH);
  Pipeline build();

  struct Group {
    std::string module;
    std::string entry;
  };

  struct HitGroup {
    Group closestHit;
    Group anyHit;
  };

  std::string launchParams;
  std::vector<Group> raygenGroups;
  std::vector<Group> exceptionGroups;
  std::vector<Group> missGroups;
  std::vector<HitGroup> hitGroups;
};

// the struct Context does not meant to hide all optix implementation details,
// but rather provides a cleaner wrapper for use.
struct Context {
  // should be called before any operations
  static void init();

  static OptixDeviceContext context;
  static std::string ptxDir;
};