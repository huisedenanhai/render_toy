#pragma once

#define NOMINMAX
#include "exceptions.h"
#include <iostream>
#include <map>
#include <optional>
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
  std::map<std::string, OptixModule> modules;
  OptixPipeline pipeline;

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

  void destroy();
};

// order of adding will be preserved
struct PipelineBuilder {
  struct Group {
    std::string module;
    std::string entry;
  };

  // explicitly mark unused program as null won't make the program run faster,
  // though.
  struct HitGroup {
    std::optional<Group> closestHit;
    std::optional<Group> anyHit;
  };

  PipelineBuilder &set_launch_params(const std::string &name);
  // these programs groups are supposed to be directly mapped to the concept of
  // material
  PipelineBuilder &add_raygen_group(const std::string &module,
                                    const std::string &entry);

  PipelineBuilder &add_exception_group(const std::string &module,
                                       const std::string &entry);

  PipelineBuilder &add_miss_group(const std::string &module,
                                  const std::string &entry);

  PipelineBuilder &add_hit_group(HitGroup hitGroup);

  Pipeline build();

private:
  std::string _launchParams;
  std::vector<Group> _raygenGroups;
  std::vector<Group> _exceptionGroups;
  std::vector<Group> _missGroups;
  std::vector<HitGroup> _hitGroups;
};

template <typename T> struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) Record {
  char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  T data;
};

struct ShaderBindingTable {
  // device memory buffers
  void *sbtBuffers_d[Pipeline::groupCnt]{};
  // size in bytes
  size_t sbtBufferSizes[Pipeline::groupCnt]{};
  // stride in bytes
  size_t sbtBufferStrides[Pipeline::groupCnt]{};
  OptixShaderBindingTable sbt{};

  CUdeviceptr calculate_buffer_offset(unsigned int group, unsigned int index);

  // use the first raygen record by default
  ShaderBindingTable &set_raygen_record(unsigned int index);

  // use the first exception record by default
  ShaderBindingTable &set_exception_record(unsigned int index);

  void destroy();
};

struct IDeleter {
  virtual void destroy(void *ptr) = 0;
  virtual ~IDeleter() {}
};

template <typename T> struct Deleter : public IDeleter {
  virtual void destroy(void *ptr) override {
    delete (T *)ptr;
  }
};

struct GenericDeleter {
  template <typename T>
  explicit GenericDeleter(std::unique_ptr<Deleter<T>> d)
      : deleter(std::move(d)) {}
  GenericDeleter() : deleter(nullptr) {}
  void operator()(void *ptr) {
    if (deleter && ptr) {
      deleter->destroy(ptr);
    }
  }
  std::unique_ptr<IDeleter> deleter;
};

struct ShaderBindingTableBuilder {
  struct RecordHandle {
    RecordHandle() : record(nullptr, GenericDeleter()), size(0) {}
    template <typename T>
    RecordHandle(T *ptr)
        : record(ptr, GenericDeleter(std::make_unique<Deleter<T>>())),
          size(sizeof(T)) {}
    RecordHandle(const RecordHandle &) = delete;
    RecordHandle(RecordHandle &&) = default;
    RecordHandle &operator=(const RecordHandle &) = delete;
    RecordHandle &operator=(RecordHandle &&) = default;
    ~RecordHandle() = default;

    std::unique_ptr<void, GenericDeleter> record;
    // size in bytes
    size_t size;
  };

  // the pipeline must be fully built
  ShaderBindingTableBuilder(Pipeline *pipeline) {
    this->pipeline = pipeline;
    for (unsigned int i = 0; i < Pipeline::groupCnt; i++) {
      records[i].resize(0);
    }
  }
  ShaderBindingTableBuilder(const ShaderBindingTableBuilder &) = delete;
  ShaderBindingTableBuilder(ShaderBindingTableBuilder &&) = default;
  ShaderBindingTableBuilder &
  operator=(const ShaderBindingTableBuilder &) = delete;
  ShaderBindingTableBuilder &operator=(ShaderBindingTableBuilder &&) = default;

  ~ShaderBindingTableBuilder() = default;

  template <typename T>
  T *add_record(unsigned int groupIndex, unsigned int index) {
    auto record = new Record<T>;
    TOY_OPTIX_CHECK_OR_THROW(
        optixSbtRecordPackHeader(pipeline->groups[groupIndex][index],
                                 record), );

    records[groupIndex].emplace_back(record);
    return &record->data;
  }

  void add_record_no_data(unsigned int groupIndex, unsigned int index) {
    struct NoData {};
    auto record = new Record<NoData>;
    TOY_OPTIX_CHECK_OR_THROW(
        optixSbtRecordPackHeader(pipeline->groups[groupIndex][index],
                                 record), );

    records[groupIndex].emplace_back(record);
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

  void add_raygen_record_no_data(unsigned int index) {
    add_record_no_data(Pipeline::raygenGroupIndex, index);
  }

  void add_miss_record_no_data(unsigned int index) {
    add_record_no_data(Pipeline::missGroupIndex, index);
  }

  void add_exception_record_no_data(unsigned int index) {
    add_record_no_data(Pipeline::exceptionGroupIndex, index);
  }

  void add_hit_record_no_data(unsigned int index) {
    add_record_no_data(Pipeline::hitGroupIndex, index);
  }

  size_t get_hit_record_count() {
    return records[Pipeline::hitGroupIndex].size();
  }

  ShaderBindingTable build();

  std::vector<RecordHandle> records[Pipeline::groupCnt];
  Pipeline *pipeline;
};

struct BoundingSphere {
  float3 center;
  float radius;
};

struct AABB {
  float3 min;
  float3 max;

  AABB extend(const float3 &p);
  // extend in place
  void extend_(const float3 &p);
  BoundingSphere get_bouding_sphere();
};

struct GAS {
  OptixTraversableHandle gas = 0;
  void *accel_d = nullptr;
  void *vertices_d = nullptr;
  void *indices_d = nullptr;
  void *materialIds_d = nullptr;
  AABB aabb;

  void destroy();
};

struct GASBuilder {
  GAS build();
  AABB calculate_aabb();

  // all these arrays are packed
  float *vertices;
  // how many triples are in the vertices buffer
  unsigned int vertexCount;
  unsigned int *indices;
  // with size of primitive count, stores offset of hit records in sbt
  unsigned int *materialIds;
  unsigned int primitiveCount;
  // how many hit records in sbt
  unsigned int materialCount;
};

// the struct Context does not meant to hide all optix implementation details,
// but rather provides a cleaner wrapper for use.
struct Context {
  // should be called before any operations
  static void init();

  static OptixDeviceContext context;
  static std::string ptxDir;
};