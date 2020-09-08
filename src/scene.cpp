#include "scene.h"
#include "rotation.h"
#include <cpptoml/cpptoml.h>
#include <iostream>
#include <unordered_map>

std::unique_ptr<RGB2Spectral> Scene::rgb2spectral = nullptr;

template <typename T>
inline T get_value_required(const std::shared_ptr<cpptoml::table> &table,
                            const std::string &key,
                            const std::string &tableName) {
  auto v = table->get_as<T>(key);
  if (!v) {
    throw std::runtime_error("value for \"" + tableName + "." + key +
                             "\" is required");
  }
  return *v;
}

inline float3 get_float3(const std::shared_ptr<cpptoml::table> &table,
                         const std::string &key,
                         const float3 &defaultValue) {
  auto value = table->get_array_of<double>(key);
  if (value && value->size() == 3) {
    return make_float3(value->at(0), value->at(1), value->at(2));
  }
  return defaultValue;
}

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
  camera.position = get_float3(table, "position", make_float3(0, 0, 0));
  auto eulerXYZ = get_float3(table, "rotation", make_float3(0, 0, 0));
  auto rotation = euler_xyz_to_matrix(eulerXYZ.x, eulerXYZ.y, eulerXYZ.z);
  camera.right = make_float3(rotation[0][0], rotation[0][1], rotation[0][2]);
  camera.up = make_float3(rotation[1][0], rotation[1][1], rotation[1][2]);
  camera.back = make_float3(rotation[2][0], rotation[2][1], rotation[2][2]);
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
  frame.hdr = table->get_as<bool>("hdr").value_or(frame.hdr);
  frame.exposure = table->get_as<double>("exposure").value_or(frame.exposure);
  frame.denoise = table->get_as<bool>("denoise").value_or(frame.denoise);
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

namespace ma {
struct MeshData {
  std::vector<float> vertices;
  std::vector<unsigned int> indices;
};

std::istream &skip_white_space(std::istream &is) {
  char cur = is.get();
  while (isspace(cur)) {
    cur = is.get();
  }
  if (is.good()) {
    is.unget();
  }
  return is;
}

std::string get_token(std::istream &is) {
  constexpr char oneCharToken[] = {'=', ',', '[', ']'};
  auto isOneCharToken = [&](char c) {
    for (auto t : oneCharToken) {
      if (c == t) {
        return true;
      }
    }
    return false;
  };
  skip_white_space(is);
  char cur = is.get();
  if (isOneCharToken(cur)) {
    char str[] = {cur, 0};
    return str;
  }

  std::stringstream ss;
  while (is && !isOneCharToken(cur) && !isspace(cur)) {
    ss << cur;
    cur = is.get();
  }
  if (isOneCharToken(cur)) {
    is.unget();
  }
  return ss.str();
}

void expect_token(const std::string &token, const std::string &expect) {
  if (token != expect) {
    throw std::runtime_error("failed to load ma: expect " + expect +
                             " but got " + token);
  }
}

void skip_list(std::ifstream &is) {
  expect_token(get_token(is), "=");
  expect_token(get_token(is), "[");
  while (is && get_token(is) != "]") {
  }
}

template <typename T> struct from_string;
template <> struct from_string<float> {
  float operator()(const std::string &str) {
    return std::stof(str);
  }
};

template <> struct from_string<unsigned int> {
  unsigned int operator()(const std::string &str) {
    return std::stoul(str);
  }
};

template <typename T> std::vector<T> load_list(std::ifstream &is) {
  expect_token(get_token(is), "=");
  expect_token(get_token(is), "[");
  std::vector<T> res;
  while (is) {
    auto token = get_token(is);
    if (token == "]") {
      break;
    }
    res.push_back(from_string<T>{}(token));
    expect_token(get_token(is), ",");
  }
  return res;
}

// we have to implement loading logic of ma file as cpptoml may complain the
// list is not homogenious, and it handles memory not efficient enough for large
// data.
// this method can be called in parallel as long as paths are different
MeshData load_ma(const fs::path &path) {
  std::ifstream is(path);
  MeshData data;
  while (is) {
    auto key = get_token(is);
    if (key == "") {
      break;
    }
    if (key == "vertices") {
      data.vertices = load_list<float>(is);
    } else if (key == "indices") {
      data.indices = load_list<unsigned int>(is);
    } else {
      skip_list(is);
    }
  }
  is.close();
  return data;
}
} // namespace ma

inline void build_sbt(Scene &scene,
                      IIntegrator &integrator,
                      const std::shared_ptr<cpptoml::table> &toml) {
  auto builder = integrator.get_stb_builder(toml);
  // always add a default material
  auto &defaultMat = integrator.materials.at("default");
  defaultMat.material->add_hit_record(builder, defaultMat.index, nullptr);
  scene.materialIndices["default"] = 0;
  unsigned int materialIndex = 1;
  auto mats = toml->get_table_array("material");
  if (mats) {
    for (auto data : *mats) {
      auto id = get_value_required<std::string>(data, "id", "material");
      auto type = data->get_as<std::string>("type").value_or("default");
      auto matIt = integrator.materials.find(type);
      auto &mat =
          matIt == integrator.materials.end() ? defaultMat : matIt->second;
      mat.material->add_hit_record(builder, mat.index, data);
      scene.materialIndices[id] = materialIndex++;
    }
  }
  scene.sbt = builder.build();
}

// same hash from glm
namespace {
void hash_combine(size_t &seed, size_t hash) {
  hash += 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= hash;
}
} // namespace

namespace std {
template <> struct hash<float3> {
  size_t operator()(const float3 &v) const {
    size_t seed = 0;
    hash<float> hasher;
    hash_combine(seed, hasher(v.x));
    hash_combine(seed, hasher(v.y));
    hash_combine(seed, hasher(v.z));
    return seed;
  }
};
} // namespace std

bool operator==(const float3 &lhs, const float3 &rhs) {
  return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
}

inline void build_gas(Scene &scene,
                      const std::shared_ptr<cpptoml::table> &toml) {
  auto meshes = toml->get_table_array("mesh");
  if (!meshes) {
    throw std::runtime_error("scene file has no mesh data");
  }
  std::unordered_map<float3, uint32_t> uniqueVertices;
  std::vector<float> vertices;
  std::vector<unsigned int> indices;
  std::vector<unsigned int> materialIds;
  unsigned int matCount = 0;
  unsigned int rawVertexCount = 0;
  for (auto &mesh : *meshes) {
    auto meshPath =
        fs::path(mesh->get_as<std::string>(parentDirKey).value_or(""));
    meshPath /= mesh->get_as<std::string>("file").value_or("");
    auto meshData = ma::load_ma(meshPath);
    rawVertexCount += meshData.vertices.size() / 3;
    for (auto index : meshData.indices) {
      float3 vertex = make_float3(meshData.vertices[3 * index],
                                  meshData.vertices[3 * index + 1],
                                  meshData.vertices[3 * index + 2]);
      unsigned int modifiedIndex = 0;
      if (uniqueVertices.count(vertex) == 0) {
        modifiedIndex = vertices.size() / 3;
        uniqueVertices[vertex] = modifiedIndex;
        vertices.push_back(vertex.x);
        vertices.push_back(vertex.y);
        vertices.push_back(vertex.z);
      } else {
        modifiedIndex = uniqueVertices[vertex];
      }
      indices.push_back(modifiedIndex);
    }
    auto matId = scene.materialIndices[mesh->get_as<std::string>("material")
                                           .value_or("default")];
    matCount = std::max(matId + 1, matCount);
    for (int i = 0; i < meshData.indices.size() / 3; i++) {
      materialIds.push_back(matId);
    }
  }
  std::cout << "raw vertex count: " << rawVertexCount
            << ", unique vertex count: " << uniqueVertices.size() << std::endl;
  GASBuilder builder;
  builder.vertices = &vertices[0];
  builder.vertexCount = vertices.size() / 3;
  builder.indices = &indices[0];
  builder.materialIds = &materialIds[0];
  builder.primitiveCount = indices.size() / 3;
  builder.materialCount = matCount;
  scene.gas = builder.build();
}

Scene SceneLoader::load(const fs::path &sceneToml, IIntegrator &integrator) {
  auto toml = load_toml(sceneToml);
  Scene scene;
  load_launch(scene.launch, toml->get_table("launch"));
  load_tile(scene.tile, toml->get_table("tile"));
  load_frame(scene.frame, toml->get_table("frame"));
  load_camera(scene.camera, toml->get_table("camera"));
  build_sbt(scene, integrator, toml);
  build_gas(scene, toml);
  return scene;
}