#include <tinyobj\tiny_obj_loader.h>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <vector>

namespace fs = std::filesystem;
constexpr float Pi = 3.1415926f;

struct Config {
  fs::path objFile;
  fs::path outputDir;
} g_Config;

struct SubShape {
  std::vector<float> vertices;
  std::vector<unsigned int> indices;
};

struct Material {
  std::string id;
  float baseColor[3];
  float emission[3];
};

struct Color {
  const float *color;
};

Color color(const float *cv) {
  Color c;
  c.color = cv;
  return c;
}

std::ostream &operator<<(std::ostream &os, const Color &c) {
  os << "[" << std::fixed << std::setprecision(6) << c.color[0] << ", "
     << c.color[1] << ", " << c.color[2] << "]";
  return os;
}

std::ostream &operator<<(std::ostream &os, const Material &mat) {
  os << "[[material]]" << std::endl;
  os << "id = " << std::quoted(mat.id) << std::endl;
  os << "baseColor = " << color(mat.baseColor) << std::endl;
  os << "emission = " << color(mat.emission) << std::endl;
  return os;
}

struct Mesh {
  std::string file;
  std::string materialId;
};

std::ostream &operator<<(std::ostream &os, const Mesh &mesh) {
  os << "[[mesh]]" << std::endl;
  os << "file = " << std::quoted(mesh.file) << std::endl;
  os << "material = " << std::quoted(mesh.materialId) << std::endl;
  return os;
}

inline char value_to_hex(uint8_t v) {
  assert(v >= 0 && v < 16);
  if (v < 10)
    return '0' + v;
  return 'a' + v - 10;
}

inline uint8_t fetch_first(uint8_t v) {
  return (v >> 4) & 0xF;
}

inline uint8_t fetch_second(uint8_t v) {
  return v & 0xF;
}

struct Guid {
  static Guid create();
  std::string to_string() const;
  bool is_nil();
  alignas(8) uint8_t data[16]{};
};

bool Guid::is_nil() {
  for (auto d : data) {
    if (d != 0) {
      return false;
    }
  }
  return true;
}

std::string Guid::to_string() const {
  // 8-4-4-4-12
  std::string str(36, 0);
  int j = 0;
  int i = 0;
  int indices[4] = {4, 6, 8, 10};
  for (int k = 0; k < 4; k++) {
    int high = indices[k];
    for (; i < high; i++) {
      str[j++] = value_to_hex(fetch_first(data[i]));
      str[j++] = value_to_hex(fetch_second(data[i]));
    }

    str[j++] = '-';
  }
  for (; i < 16; i++) {
    str[j++] = value_to_hex(fetch_first(data[i]));
    str[j++] = value_to_hex(fetch_second(data[i]));
  }
  return str;
}

Guid Guid::create() {
  static bool first = true;
  static std::mt19937_64 mt_rand;
  if (first) {
    auto seed =
        std::chrono::high_resolution_clock::now().time_since_epoch().count();
    mt_rand.seed(seed);
    first = false;
  }
  Guid guid{};
  Guid nil{};
  while (guid.is_nil()) {
    auto d = reinterpret_cast<uint64_t *>(guid.data);
    d[0] = mt_rand();
    d[1] = mt_rand();
  }
  return guid;
}

std::string gen_unique_name(const std::string &name) {
  auto guid = Guid::create().to_string();
  if (name == "") {
    return guid;
  }
  if (name[name.size() - 1] != '.') {
    return name + "." + guid;
  }
  return name + guid;
}

struct Model {
  std::vector<Material> materials;
  std::vector<Mesh> meshes;
};

template <typename T>
std::ostream &write_list(std::ostream &os,
                         const std::string &name,
                         const std::vector<T> &data) {
  std::string indent = "  ";
  os << name << " = [" << std::endl;
  os << indent;
  auto dataCnt = data.size();
  for (unsigned int i = 0; i < dataCnt; i++) {
    os << data[i] << ", ";
    if (i % 3 == 2) {
      os << std::endl;
      if (i != dataCnt - 1) {
        os << indent;
      }
    }
  }
  os << "]" << std::endl;
  return os;
}

void write_ma(const fs::path &filename,
              const std::vector<float> &vertices,
              const std::vector<unsigned int> &indices) {
  std::ofstream os(filename);
  write_list(os, "vertices", vertices);
  os << std::endl;
  write_list(os, "indices", indices);
  os.close();
}

void write_model(const fs::path &filename, const Model &model) {
  std::ofstream os(filename);
  for (const auto &material : model.materials) {
    os << material << std::endl;
  }
  for (const auto &mesh : model.meshes) {
    os << mesh << std::endl;
  }
  os.close();
}

bool parse_args(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "please specify input file" << std::endl;
    return false;
  }
  g_Config.objFile = argv[1];
  g_Config.objFile.make_preferred();
  if (argc < 3) {
    std::cout << "no output dir specified, use current working directory"
              << std::endl;
  }
  g_Config.outputDir = argc < 3 ? fs::current_path() : argv[2];
  g_Config.outputDir.make_preferred();
  return true;
}

int main(int argc, char **argv) {
  if (!parse_args(argc, argv)) {
    return -1;
  }

  // load obj
  tinyobj::attrib_t attrib;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;

  {
    auto inputFile = g_Config.objFile.string();
    std::cout << "start load obj file: " << inputFile << std::endl;
    char preferedSeperator = fs::path::preferred_separator;
    auto materialBaseDir =
        g_Config.objFile.parent_path().string() + preferedSeperator;
    std::string loadError;
    if (!tinyobj::LoadObj(&attrib,
                          &shapes,
                          &materials,
                          &loadError,
                          inputFile.c_str(),
                          materialBaseDir.c_str(),
                          true)) {
      std::cerr << "failed to load obj file: " << loadError << std::endl;
      return -1;
    } else {
      std::cout << "finish load obj file " << inputFile;
      if (!loadError.empty()) {
        std::cout << " with info:\n" << loadError;
      }
      std::cout << std::endl;
    }
  }
  std::string componentsFolder = "components";
  auto componetsDir = g_Config.outputDir;
  componetsDir /= componentsFolder;
  try {
    fs::create_directories(componetsDir);
  } catch (std::exception &e) {
    std::cerr << "failed to create component directory " << componetsDir
              << std::endl;
    return -1;
  }
  auto modelPath = g_Config.outputDir;
  modelPath /= g_Config.objFile.filename().replace_extension(".toml");

  auto getCompPath = [&](const auto &name) {
    auto n = componetsDir;
    n /= fs::path(name).replace_extension(".ma");
    return n;
  };

  auto getCompPathRelative = [&](const auto &name) {
    auto n = fs::path(componentsFolder);
    n /= fs::path(name).replace_extension(".ma");
    return n;
  };

  // it's possible for the material count to be zero
  if (materials.size() == 0) {
    // write a mesh with default material
    auto compPath = getCompPath(g_Config.objFile.filename());
    // concat all indices
    size_t indicesCnt = 0;
    for (auto &shape : shapes) {
      indicesCnt += shape.mesh.indices.size();
    }
    std::vector<unsigned int> indices{};
    indices.reserve(indicesCnt);
    for (auto &shape : shapes) {
      for (auto index : shape.mesh.indices) {
        indices.push_back(index.vertex_index);
      }
    }
    write_ma(compPath, attrib.vertices, indices);
    // write model file
    Model model;
    Material mat;
    mat.baseColor[0] = 0.7f;
    mat.baseColor[1] = 0.7f;
    mat.baseColor[2] = 0.7f;
    mat.emission[0] = 1.0f;
    mat.emission[1] = 1.0f;
    mat.emission[2] = 1.0f;
    mat.id = gen_unique_name(
        g_Config.objFile.filename().replace_extension().string() + ".default");
    model.materials.push_back(mat);
    Mesh mesh;
    mesh.materialId = mat.id;
    mesh.file = getCompPathRelative(g_Config.objFile.filename()).string();
    model.meshes.push_back(mesh);
    write_model(modelPath, model);
  } else {
    Model model;
    model.materials.reserve(materials.size());
    for (auto &mat : materials) {
      Material material;
      memcpy(material.baseColor, mat.diffuse, 3 * sizeof(float));
      memcpy(material.emission, mat.emission, 3 * sizeof(float));
      // scale emission to let it use radiance as unit
      for (int i = 0; i < 3; i++) {
        material.emission[i] *= Pi;
      }
      material.id = gen_unique_name(mat.name);
      model.materials.push_back(material);
    }
    // split each shape by material indices
    for (const auto &shape : shapes) {
      std::map<int, std::vector<unsigned int>> indices;
      auto fc = shape.mesh.material_ids.size();
      assert(fc == shape.mesh.indices.size() / 3);
      for (unsigned int i = 0; i < fc; i++) {
        auto matId = shape.mesh.material_ids[i];
        indices[matId].push_back(shape.mesh.indices[3 * i].vertex_index);
        indices[matId].push_back(shape.mesh.indices[3 * i + 1].vertex_index);
        indices[matId].push_back(shape.mesh.indices[3 * i + 2].vertex_index);
      }
      for (auto &p : indices) {
        auto compName = gen_unique_name(shape.name) + ".ma";
        auto compPath = getCompPath(compName);
        std::set<unsigned int> uniqueIndices(p.second.begin(), p.second.end());
        std::map<unsigned int, unsigned int> indexMapping;
        std::vector<float> filteredVertices;
        std::vector<unsigned int> filteredIndices;
        auto uniqueIndexCount = uniqueIndices.size();
        filteredVertices.reserve(3 * uniqueIndexCount);
        filteredIndices.reserve(p.second.size());
        for (auto index : uniqueIndices) {
          indexMapping[index] = filteredVertices.size() / 3;
          for (int i = 0; i < 3; i++) {
            filteredVertices.push_back(attrib.vertices[3 * index + i]);
          }
        }
        for (auto index : p.second) {
          filteredIndices.push_back(indexMapping[index]);
        }
        write_ma(compPath, filteredVertices, filteredIndices);
        Mesh mesh;
        mesh.file = getCompPathRelative(compName).string();
        mesh.materialId = model.materials[p.first].id;
        model.meshes.push_back(mesh);
      }
    }
    write_model(modelPath, model);
  }
  std::cout << "output written to " << modelPath << " successfully"
            << std::endl;
  return 0;
}