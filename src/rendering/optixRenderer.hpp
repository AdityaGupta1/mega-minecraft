#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <optix.h>
#include <optix_stubs.h>

#include <vector>

#include "cuda/cudaUtils.hpp"
#include "player/player.hpp"
#include "terrain/terrain.hpp"
#include "util/common.h"

class OptixRenderer
{
public:
    OptixRenderer(Terrain* terrain);

protected:
    GLFWwindow* window{ nullptr };
    Terrain* terrain{ nullptr };

    CUcontext          cudaContext = {};
    CUstream           stream;

    OptixDeviceContext optixContext = {};

    OptixPipeline               pipeline;
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    OptixPipelineLinkOptions    pipelineLinkOptions = {};

    OptixModule                 module;
    OptixModuleCompileOptions   moduleCompileOptions = {};

    struct {
        OptixTraversableHandle handle;
        CUBuffer outputBuffer;
        CUBuffer tempBuffer;
        OptixBuildInput buildInput;
        OptixAccelBuildOptions buildOptions;
        OptixAccelBufferSizes bufferSizes;
    } rootIAS;

    CUBuffer chunkInstancesBuffer;
    std::vector<OptixInstance> chunkInstances;

    std::vector<OptixProgramGroup> raygenProgramGroups;
    std::vector<OptixProgramGroup> missProgramGroups;
    std::vector<OptixProgramGroup> hitProgramGroups;

    OptixShaderBindingTable sbt = {};

    void createContext();
    void buildChunkAccel(const Chunk* c);
    void buildRootAccel();
    void createModule();
    void createProgramGroups();
    void createPipeline();
    void buildSBT();
    void optixRenderFrame();
};
