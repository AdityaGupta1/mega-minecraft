#include "optixRenderer.hpp"
#include <optix_function_table_definition.h>

OptixRenderer::OptixRenderer()
{
    createContext();
}

void OptixRenderer::createContext()
{
    cudaFree(0);
    optixInit();
    OptixDeviceContextOptions options = {};
    optixDeviceContextCreate(cudaContext, &options, &optixContext);
}

void OptixRenderer::buildAccel()
{
}

void OptixRenderer::createModule()
{
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions.numPayloadValues = 2;
    pipelineCompileOptions.numAttributeValues = 2;
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";

    pipelineLinkOptions.maxTraceDepth = 1;

    // optixCreateModule here but need ptx / optixir stuff
}

void OptixRenderer::createProgramGroups()
{
}

void OptixRenderer::createPipeline()
{
}

void OptixRenderer::buildSBT()
{
}
