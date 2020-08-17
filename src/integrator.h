#pragma once
#include "context.h"
#include <cpptoml/cpptoml.h>
#include <map>
#include <string>

class IMaterial {
public:
  // add one and only one record to the STB. add a default record if data is
  // null
  virtual void add_hit_record(ShaderBindingTableBuilder &builder,
                              unsigned int index,
                              const std::shared_ptr<cpptoml::table> &data) = 0;

  virtual ~IMaterial() {}
};

class IIntegrator {
public:
  struct MaterialEntry {
    unsigned int index;
    std::unique_ptr<IMaterial> material;
  };
  Pipeline pipeline;
  // there should be at least one material with the name "default"
  std::map<std::string, MaterialEntry> materials;

  // a partially inited sbt builder, only hit records are untouched
  virtual ShaderBindingTableBuilder
  get_stb_builder(const std::shared_ptr<cpptoml::table> &toml) = 0;

  virtual ~IIntegrator() {}
};

class IIntegratorBuilder {
public:
  virtual std::unique_ptr<IIntegrator> build() = 0;
};