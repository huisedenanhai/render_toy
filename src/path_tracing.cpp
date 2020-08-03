#include "path_tracing.h"
#include "pipeline.h"

inline float3 get_color(const std::shared_ptr<cpptoml::table> &data,
                        const std::string &key,
                        float3 defaultValue) {
  if (!data) {
    return defaultValue;
  }
  auto v = data->get_array_of<double>(key);
  if (!v) {
    return defaultValue;
  }
  return make_float3(v->at(0), v->at(1), v->at(2));
}

void PathDefaultMaterial::add_hit_record(
    ShaderBindingTableBuilder &builder,
    unsigned int index,
    const std::shared_ptr<cpptoml::table> &data) {
  auto record = builder.add_hit_record<dev::HitGroupData>(index);
  record->baseColor =
      get_color(data, "baseColor", make_float3(0.7f, 0.7f, 0.7f));
  record->emission = get_color(data, "emission", make_float3(0.0f, 0.0f, 0.0f));
}

ShaderBindingTableBuilder PathIntegrator::get_stb_builder() {
  ShaderBindingTableBuilder builder(&pipeline);
  builder.add_raygen_record<dev::RayGenData>(0);
  auto exceptionRecord = builder.add_exception_record<dev::ExceptionData>(0);
  exceptionRecord->errorColor = make_float3(0, 1.0f, 0);
  auto missRecord = builder.add_miss_record<dev::MissData>(0);
  missRecord->color = make_float3(0.0f, 0.0f, 0.0f);
  return builder;
}

std::unique_ptr<IIntegrator> PathIntegratorBuilder::build() {
  auto res = std::make_unique<PathIntegrator>();
  auto moduleName = "pipeline.ptx";
  res->pipeline =
      PipelineBuilder()
          .set_launch_params("g_LaunchParams")
          .add_raygen_group(moduleName, "__raygen__entry")
          .add_miss_group(moduleName, "__miss__entry")
          .add_exception_group(moduleName, "__exception__entry")
          .add_hit_group(
              moduleName, "__closesthit__entry", moduleName, "__anyhit__entry")
          .build();

  auto &defaultMat = res->materials["default"];
  defaultMat.material = std::make_unique<PathDefaultMaterial>();
  defaultMat.index = 0;

  return res;
}