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
  };

  struct Tile {
    unsigned int width = 64;
    unsigned int height = 64;
  };

  struct Camera {
    float3 position{};
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