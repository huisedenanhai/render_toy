#include "path_tracing.h"
#include "pipeline.h"

inline float3 get_color(const std::shared_ptr<cpptoml::table> &data,
                        const std::string &key,
                        float3 defaultValue) {
  if (!data) {
    return defaultValue;
  }
  auto v = data->get_qualified_array_of<double>(key);
  if (!v) {
    return defaultValue;
  }
  return make_float3(v->at(0), v->at(1), v->at(2));
}

void PathDiffuseMaterial::add_hit_record(
    ShaderBindingTableBuilder &builder,
    unsigned int index,
    const std::shared_ptr<cpptoml::table> &data) {
  auto record = builder.add_hit_record<dev::DiffuseHitGroupData>(index);
  record->baseColor =
      get_color(data, "baseColor", make_float3(0.7f, 0.7f, 0.7f));
  record->emission = get_color(data, "emission", make_float3(0.0f, 0.0f, 0.0f));
}

void PathGlassMaterial::add_hit_record(
    ShaderBindingTableBuilder &builder,
    unsigned int index,
    const std::shared_ptr<cpptoml::table> &data) {
  auto record = builder.add_hit_record<dev::GlassHitGroupData>(index);
  record->ior = data->get_as<double>("ior").value_or(1.3);
  record->baseColor =
      get_color(data, "baseColor", make_float3(0.7f, 0.7f, 0.7f));
}

ShaderBindingTableBuilder PathIntegrator::get_stb_builder(const std::shared_ptr<cpptoml::table> &toml) {
  ShaderBindingTableBuilder builder(&pipeline);
  builder.add_raygen_record<dev::RayGenData>(0);
  auto exceptionRecord = builder.add_exception_record<dev::ExceptionData>(0);
  exceptionRecord->errorColor = make_float3(0, 1.0f, 0);
  auto missRecord = builder.add_miss_record<dev::MissData>(0);
  missRecord->color = get_color(toml, "miss.color", make_float3(0.5f, 0.5f, 0.5f));
  return builder;
}

std::unique_ptr<IIntegrator> PathIntegratorBuilder::build() {
  auto res = std::make_unique<PathIntegrator>();
  auto moduleName = "pipeline.ptx";

  struct HitGroupEntry {
    HitGroupEntry(std::string n,
                  std::string eCH,
                  std::string eAH,
                  std::unique_ptr<IMaterial> mat)
        : name(std::move(n)), entryCH(std::move(eCH)), entryAH(std::move(eAH)),
          material(std::move(mat)) {}
    std::string name;
    std::string entryCH;
    std::string entryAH;
    std::unique_ptr<IMaterial> material;
  };

  std::vector<HitGroupEntry> hitGroups;
  hitGroups.emplace_back("default",
                         "__closesthit__diffuse",
                         "__anyhit__diffuse",
                         std::make_unique<PathDiffuseMaterial>());
  hitGroups.emplace_back("glass",
                         "__closesthit__glass",
                         "__anyhit__glass",
                         std::make_unique<PathGlassMaterial>());

  auto builder = PipelineBuilder()
                     .set_launch_params("g_LaunchParams")
                     .add_raygen_group(moduleName, "__raygen__entry")
                     .add_miss_group(moduleName, "__miss__entry")
                     .add_exception_group(moduleName, "__exception__entry");
  // add materials
  for (unsigned int i = 0; i < hitGroups.size(); i++) {
    auto &group = hitGroups[i];
    builder.add_hit_group(moduleName, group.entryCH, moduleName, group.entryAH);
    res->materials[group.name].material = std::move(group.material);
    res->materials[group.name].index = i;
  }

  res->pipeline = builder.build();
  return res;
}