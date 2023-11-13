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
    OptixRenderer(GLFWwindow* window, ivec2* windowSize, Terrain* terrain, Player* player);

protected:
    GLFWwindow* window{ nullptr };
    ivec2* windowSize{ nullptr };
    Terrain* terrain{ nullptr };
    Player* player{ nullptr };

    CUcontext          cudaContext = {};
    CUstream           stream;

    OptixDeviceContext optixContext = {};

    OptixPipeline               pipeline;
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    OptixPipelineLinkOptions    pipelineLinkOptions = {};

    OptixModule                 module;
    OptixModuleCompileOptions   moduleCompileOptions = {};

    std::vector<Texture> textures;
    std::vector<cudaArray_t> texArrays;
    std::vector<cudaTextureObject_t> texObjects;

    struct {
        OptixTraversableHandle handle;
        CUBuffer outputBuffer;
        CUBuffer tempBuffer;
        OptixBuildInput buildInput;
        OptixAccelBuildOptions buildOptions;
        OptixAccelBufferSizes bufferSizes;
    } rootIAS;

    CUBuffer chunkInstancesBuffer;
    std::vector<OptixInstance> chunkInstances = {};
    
    std::vector<CUBuffer> vertexBuffer;
    std::vector<CUBuffer> indexBuffer;

    std::vector<OptixProgramGroup> raygenProgramGroups;
    std::vector<OptixProgramGroup> missProgramGroups;
    std::vector<OptixProgramGroup> hitProgramGroups;

    CUBuffer raygenRecordBuffer;
    CUBuffer missRecordBuffer;
    CUBuffer hitRecordBuffer;
    OptixShaderBindingTable sbt = {};

    CUBuffer playerInfoBuffer;

    void createContext();
    void createTextures();
    void buildChunkAccel(const Chunk* c);
    void buildRootAccel();
    void createModule();
    void createProgramGroups();
    void createPipeline();
    void buildSBT();
    void optixRenderFrame();
};
