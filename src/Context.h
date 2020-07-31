#pragma once

#define NOMINMAX
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
  constexpr static int raygenGroupIndex = 0;
  constexpr static int missGroupIndex = 1;
  constexpr static int exceptionGroupIndex = 2;
  constexpr static int hitGroupIndex = 3;
  constexpr static int groupCnt = 4;

  std::vector<OptixProgramGroup> groups[groupCnt];

  inline std::vector<OptixProgramGroup> &raygenGroups() {
    return groups[raygenGroupIndex];
  }

  inline std::vector<OptixProgramGroup> &missGroups() {
    return groups[missGroupIndex];
  }

  inline std::vector<OptixProgramGroup> &exceptionGroups() {
    return groups[exceptionGroupIndex];
  }

  inline std::vector<OptixProgramGroup> &hitGroups() {
    return groups[hitGroupIndex];
  }

  std::map<std::string, OptixModule> modules;
  OptixPipeline pipeline;
};

// order of adding will be preserved
struct PipelineBuilder {
  PipelineBuilder &set_launch_params(const std::string &name);
  // these programs groups are supposed to be directly mapped to the concept of
  // material
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

template <typename T> struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) Record {
  char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  T data;
};

struct ShaderBindingTable {
  struct RecordHandle {
    void *record;
    size_t size;

    void destroy() {
      delete record;
    }
  };

  // the pipeline must be fully built
  ShaderBindingTable(Pipeline *pipeline) {
    this->pipeline = pipeline;
    for (unsigned int i = 0; i < Pipeline::groupCnt; i++) {
      records[i].resize(0);
    }
  }
  ShaderBindingTable(const ShaderBindingTable &) = delete;
  ShaderBindingTable(ShaderBindingTable &&) = delete;
  ShaderBindingTable &operator=(const ShaderBindingTable &) = delete;
  ShaderBindingTable &operator=(ShaderBindingTable &&) = delete;

  ~ShaderBindingTable();

  template <typename T>
  T *add_record(unsigned int groupIndex, unsigned int index) {
    auto record = new Record<T>;
    TOY_OPTIX_CHECK_OR_THROW(
        optixSbtRecordPackHeader(pipeline->groups[groupIndex][index],
                                 record), );

    RecordHandle handle{};
    handle.record = record;
    handle.size = sizeof(Record<T>);
    records[groupIndex].push_back(handle);
    return &record->data;
  }

  template <typename T> T *add_raygen_record(unsigned int index) {
    return add_record<T>(Pipeline::raygenGroupIndex, index);
  }

  template <typename T> T *add_miss_record(unsigned int index) {
    return add_record<T>(Pipeline::missGroupIndex, index);
  }

  template <typename T> T *add_exception_record(unsigned int index) {
    return add_record<T>(Pipeline::exceptionGroupIndex, index);
  }

  template <typename T> T *add_hit_record(unsigned int index) {
    return add_record<T>(Pipeline::hitGroupIndex, index);
  }

  void commit();

  std::vector<RecordHandle> records[Pipeline::groupCnt];
  Pipeline *pipeline;
  // device memory buffers
  void *sbtBuffers_d[Pipeline::groupCnt]{};
  size_t sbtBufferSizes[Pipeline::groupCnt]{};
  size_t sbtBufferStrides[Pipeline::groupCnt]{};
  OptixShaderBindingTable sbt{};
};

// the struct Context does not meant to hide all optix implementation details,
// but rather provides a cleaner wrapper for use.
struct Context {
  // should be called before any operations
  static void init();

  static OptixDeviceContext context;
  static std::string ptxDir;
};