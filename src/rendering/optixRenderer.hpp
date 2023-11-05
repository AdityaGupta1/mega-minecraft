#pragma once

#include <optix.h>
#include <optix_stubs.h>
#include <vector>

#include "cuda/cudaUtils.hpp"
#include "player/player.hpp"
#include "terrain/terrain.hpp"

class OptixRenderer
{
public:
	OptixRenderer();

protected:
    CUcontext          cudaContext;
    CUstream           stream;

    OptixDeviceContext optixContext;

    OptixPipeline               pipeline;
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    OptixPipelineLinkOptions    pipelineLinkOptions = {};

    OptixModule                 module;
    OptixModuleCompileOptions   moduleCompileOptions = {};

    std::vector<OptixProgramGroup> raygenProgramGroups;
    std::vector<OptixProgramGroup> missProgramGroups;
    std::vector<OptixProgramGroup> hitProgramGroups;

    OptixShaderBindingTable sbt = {};

    void createContext();
    void buildAccel();
    void createModule();
    void createProgramGroups();
    void createPipeline();
    void buildSBT();
};

