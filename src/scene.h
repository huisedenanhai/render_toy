#pragma once
#include "context.h"
#include "integrator.h"
#include <filesystem>
#include <map>
#include <vector_functions.h>
#include <vector_types.h>

namespace fs = std::filesystem;

struct Scene {
  struct Launch {
    unsigned int spp = 100;
    unsigned int maxPathLength = 15;
  };

  struct Frame {
    unsigned int width = 800;
    unsigned int height = 800;
    unsigned int resolution = 1000;
    bool hdr = false;
    float exposure = 1.0f;
  };

  struct Tile {
    unsigned int width = 64;
    unsigned int height = 64;
  };

  struct Camera {
    float3 position{};
    float3 right;
    float3 up;
    float3 back;
  };

  Launch launch;
  Frame frame;
  Tile tile;
  Camera camera;
  std::map<std::string, unsigned int> materialIndices;
  ShaderBindingTable sbt;
  GAS gas;
};

struct SceneLoader {
  Scene load(const fs::path &sceneToml, IIntegrator &integrator);
};