#include "scene.h"
#include <cpptoml/cpptoml.h>
#include <iostream>

inline void load_launch(Scene::Launch &launch,
                        const std::shared_ptr<cpptoml::table> &table) {
  if (!table) {
    return;
  }
  launch.spp = table->get_as<unsigned int>("spp").value_or(launch.spp);
  launch.maxPathLength = table->get_as<unsigned int>("maxPathLength")
                             .value_or(launch.maxPathLength);
}

inline void load_tile(Scene::Tile &tile,
                      const std::shared_ptr<cpptoml::table> &table) {
  if (!table) {
    return;
  }
  tile.width = table->get_as<unsigned int>("width").value_or(tile.width);
  tile.height = table->get_as<unsigned int>("height").value_or(tile.height);
}

inline void load_camera(Scene::Camera &camera,
                        const std::shared_ptr<cpptoml::table> &table) {
  if (!table) {
    return;
  }
  auto position = table->get_array_of<double>("position");
  if (position && position->size() == 3) {
    camera.position =
        make_float3(position->at(0), position->at(1), position->at(2));
  }
}

inline void load_frame(Scene::Frame &frame,
                       const std::shared_ptr<cpptoml::table> &table) {
  if (!table) {
    return;
  }
  frame.width = table->get_as<unsigned int>("width").value_or(frame.width);
  frame.height = table->get_as<unsigned int>("height").value_or(frame.height);
  frame.resolution =
      table->get_as<unsigned int>("resolution").value_or(frame.resolution);
}

static const std::string parentDirKey = "__parent_dir";

inline std::shared_ptr<cpptoml::table> load_toml(const fs::path &path) {
  auto toml = cpptoml::parse_file(path.string());
  auto parentPath = path.parent_path().make_preferred();
  // for each element of table arrays add a field indicating parent dir
  for (auto &p : *toml) {
    if (p.second->is_table_array()) {
      for (auto &t : *p.second->as_table_array()) {
        t->insert(parentDirKey, parentPath.string());
      }
    }
  }
  // duplication or circles are not check
  auto includes = toml->get_table_array("include");
  if (includes) {
    for (auto &inc : *includes) {
      auto incPath = parentPath;
      incPath /= inc->get_as<std::string>("file").value_or("");
      auto incToml = load_toml(incPath);
      assert(incToml);
      for (auto &p : *incToml) {
        if (!toml->contains(p.first)) {
          toml->insert(p.first, p.second);
          continue;
        }
        // only try to merge table array, just to make the code simple
        if (p.second->is_table_array()) {
          auto arr = toml->get_table_array(p.first);
          if (!arr) {
            throw std::runtime_error(
                "can not merge a table array key = " + p.first + " from " +
                incPath.string() + " with no table array in " + path.string());
          }
          for (auto &t : *p.second->as_table_array()) {
            arr->insert(arr->end(), t);
          }
        } else {
          throw std::runtime_error("duplicated key " + p.first + " in file " +
                                   incPath.string() + " included by " +
                                   path.string() +
                                   ", only table array can be merged");
        }
      }
    }
  }
  return toml;
}

Scene SceneLoader::load(const fs::path &sceneToml) {
  auto toml = load_toml(sceneToml);
  Scene scene;
  load_launch(scene.launch, toml->get_table("launch"));
  load_tile(scene.tile, toml->get_table("tile"));
  load_frame(scene.frame, toml->get_table("frame"));
  load_camera(scene.camera, toml->get_table("camera"));
  for (auto &mesh : *toml->get_table_array("mesh")) {
    std::cout << mesh->get_as<std::string>("file").value_or("fuck")
              << std::endl;
    std::cout << mesh->get_as<std::string>(parentDirKey)
                     .value_or("fuck no parent dir")
              << std::endl;
  }
  return scene;
}