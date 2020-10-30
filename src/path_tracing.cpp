#include "path_tracing.h"
#include "pipeline.h"
#include "scene.h"
#include <optional>

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

inline float3 get_color_coeff(const std::shared_ptr<cpptoml::table> &data,
                              const std::string &key,
                              float3 defaultValue) {
  auto color = get_color(data, key, defaultValue);
  auto coeff = Scene::rgb2spectral->rgb_to_coeff(color);
  return coeff;
}

void PathDiffuseMaterial::add_hit_record(
    ShaderBindingTableBuilder &builder,
    unsigned int index,
    const std::shared_ptr<cpptoml::table> &data) {
  auto record = builder.add_hit_record<dev::DiffuseHitGroupData>(index);
  record->baseColorCoeff =
      get_color_coeff(data, "baseColor", make_float3(0.7f, 0.7f, 0.7f));
  record->emissionCoeff =
      get_color_coeff(data, "emission", make_float3(0.0f, 0.0f, 0.0f));
}

void PathBlackBodyMaterial::add_hit_record(
    ShaderBindingTableBuilder &builder,
    unsigned int index,
    const std::shared_ptr<cpptoml::table> &data) {
  auto record = builder.add_hit_record<dev::BlackBodyHitGroupData>(index);
  record->temperature = data->get_as<double>("temperature").value_or(6000.0);
  double strength = data->get_as<double>("strength").value_or(1.0);
  record->scaleFactor = float(strength / record->unscaled_total_energy() / 1e6);
}

void PathGlassMaterial::add_hit_record(
    ShaderBindingTableBuilder &builder,
    unsigned int index,
    const std::shared_ptr<cpptoml::table> &data) {
  auto record = builder.add_hit_record<dev::GlassHitGroupData>(index);
  auto defaultIOR = 1.45f;
  record->ior = data->get_as<double>("ior").value_or(defaultIOR);
  record->cauchy = data->get_as<double>("cauchy").value_or(0.0);
  record->baseColorCoeff =
      get_color_coeff(data, "baseColor", make_float3(0.7f, 0.7f, 0.7f));
}

ShaderBindingTableBuilder
PathIntegrator::get_stb_builder(const std::shared_ptr<cpptoml::table> &toml) {
  ShaderBindingTableBuilder builder(&pipeline);
  builder.add_raygen_record<dev::RayGenData>(0);
  auto exceptionRecord = builder.add_exception_record<dev::ExceptionData>(0);
  exceptionRecord->errorColor = make_float3(0, 1.0f, 0);
  // miss records, see pipeline.h *MissRecordIndex for order
  builder.add_miss_record_no_data(dev::ShadowMissProgramGroupIndex);
  builder.add_miss_record_no_data(dev::GeometryQueryMissProgramGroupIndex);
  auto missRecord = builder.add_miss_record<dev::MissData>(
      dev::MaterialMissProgramGroupIndex);
  missRecord->colorCoeff =
      get_color_coeff(toml, "miss.color", make_float3(0.5f, 0.5f, 0.5f));

  // hit records, see pipeline.h *HitRecordIndex for order
  builder.add_hit_record_no_data(dev::ShadowHitProgramGroupIndex);
  builder.add_hit_record_no_data(dev::GeometryQueryHitProgramGroupIndex);
  return builder;
}

std::unique_ptr<IIntegrator> PathIntegratorBuilder::build() {
  auto res = std::make_unique<PathIntegrator>();
  auto moduleName = "pipeline.ptx";

  struct HitGroupEntry {
    HitGroupEntry(std::string n,
                  std::optional<std::string> eCH,
                  std::optional<std::string> eAH,
                  std::unique_ptr<IMaterial> mat)
        : name(std::move(n)), hasCH(eCH.has_value()), entryCH(eCH.value_or("")),
          hasAH(eAH.has_value()), entryAH(eAH.value_or("")),
          material(std::move(mat)) {}
    std::string name;
    bool hasCH;
    std::string entryCH;
    bool hasAH;
    std::string entryAH;
    std::unique_ptr<IMaterial> material;
  };

  std::vector<HitGroupEntry> hitGroups;
  // built in hit groups
  hitGroups.emplace_back(
      "__shadow", "__closesthit__shadow", "__anyhit__shadow", nullptr);
  hitGroups.emplace_back("__geometry_query",
                         "__closesthit__geometry_query",
                         std::nullopt,
                         nullptr);

  // materials
  hitGroups.emplace_back("default",
                         "__closesthit__diffuse",
                         std::nullopt,
                         std::make_unique<PathDiffuseMaterial>());
  hitGroups.emplace_back("glass",
                         "__closesthit__glass",
                         std::nullopt,
                         std::make_unique<PathGlassMaterial>());
  hitGroups.emplace_back("blackbody",
                         "__closesthit__blackbody",
                         std::nullopt,
                         std::make_unique<PathBlackBodyMaterial>());

  auto builder = PipelineBuilder()
                     .set_launch_params("g_LaunchParams")
                     //.add_raygen_group(moduleName, "__raygen__path_tracing")
                     .add_raygen_group(moduleName, "__raygen__ao")
                     .add_exception_group(moduleName, "__exception__entry");

  // miss groups, order is important
  builder.add_miss_group(moduleName, "__miss__shadow");
  builder.add_miss_group(moduleName, "__miss__geometry_query");
  builder.add_miss_group(moduleName, "__miss__path_tracing");

  // add materials
  for (unsigned int i = 0; i < hitGroups.size(); i++) {
    auto &group = hitGroups[i];

    PipelineBuilder::HitGroup hg{};
    if (group.hasCH) {
      PipelineBuilder::Group ch{};
      ch.module = moduleName;
      ch.entry = group.entryCH;
      hg.closestHit = ch;
    }

    if (group.hasAH) {
      PipelineBuilder::Group ah{};
      ah.module = moduleName;
      ah.entry = group.entryAH;
    }

    builder.add_hit_group(hg);
    if (group.material) {
      res->materials[group.name].material = std::move(group.material);
      res->materials[group.name].index = i;
    }
  }

  res->pipeline = builder.build();
  return res;
}