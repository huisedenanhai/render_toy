#pragma once

#include "integrator.h"

class PathDiffuseMaterial : public IMaterial {
public:
  virtual void
  add_hit_record(ShaderBindingTableBuilder &builder,
                 unsigned int index,
                 const std::shared_ptr<cpptoml::table> &data) override;
};

class PathBlackBodyMaterial : public IMaterial {
public:
  virtual void
  add_hit_record(ShaderBindingTableBuilder &builder,
                 unsigned int index,
                 const std::shared_ptr<cpptoml::table> &data) override;
};

class PathGlassMaterial : public IMaterial {
public:
  virtual void
  add_hit_record(ShaderBindingTableBuilder &builder,
                 unsigned int index,
                 const std::shared_ptr<cpptoml::table> &data) override;
};

class PathMirrorMaterial : public IMaterial {
public:
  virtual void
  add_hit_record(ShaderBindingTableBuilder &builder,
                 unsigned int index,
                 const std::shared_ptr<cpptoml::table> &data) override;
};

class PathIsoInterferenceMaterial : public IMaterial {
public:
  virtual void
  add_hit_record(ShaderBindingTableBuilder &builder,
                 unsigned int index,
                 const std::shared_ptr<cpptoml::table> &data) override;
};

class PathIntegrator : public IIntegrator {
public:
  virtual ShaderBindingTableBuilder
  get_stb_builder(const std::shared_ptr<cpptoml::table> &toml) override;
};

class PathIntegratorBuilder : public IIntegratorBuilder {
public:
  virtual std::unique_ptr<IIntegrator> build() override;
};