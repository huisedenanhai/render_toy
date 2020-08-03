#pragma once

#include "integrator.h"

class PathDefaultMaterial : public IMaterial {
public:
  virtual void
  add_hit_record(ShaderBindingTableBuilder &builder,
                 unsigned int index,
                 const std::shared_ptr<cpptoml::table> &data) override;
};

class PathIntegrator : public IIntegrator {
public:
  virtual ShaderBindingTableBuilder get_stb_builder() override;
};

class PathIntegratorBuilder : public IIntegratorBuilder {
public:
  virtual std::unique_ptr<IIntegrator> build() override;
};