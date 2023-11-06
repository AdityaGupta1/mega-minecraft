#include "optixRenderer.hpp"
#include <optix_function_table_definition.h>

extern "C" char embedded_ptx[];

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
    createModule();
    createProgramGroups();
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

    const std::string ptxCode = embedded_ptx;
    char log[2048];
    size_t sizeof_log = sizeof(log);

    optixModuleCreate(optixContext,
        &moduleCompileOptions,
        &pipelineCompileOptions,
        ptxCode.c_str(),
        ptxCode.size(),
        log, &sizeof_log,
        &module
    );
}

void OptixRenderer::createProgramGroups()
{
    OptixProgramGroupOptions pgOptions = {};
     
    // Ray Gen
    OptixProgramGroupDesc rayGenDesc = {};
    rayGenDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    rayGenDesc.raygen.module = module;
    rayGenDesc.raygen.entryFunctionName = "__raygen__render";

    char log[2048];
    size_t sizeof_log = sizeof(log);

    optixProgramGroupCreate(
        optixContext, &rayGenDesc,
        1,  // num program groups
        &pgOptions,
        log, &sizeof_log,
        &raygenProgramGroups[0]
    );

    // Miss
    OptixProgramGroupDesc missDesc = {};
    missDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    missDesc.raygen.module = module;
    missDesc.raygen.entryFunctionName = "__miss__radiance";

    optixProgramGroupCreate(
        optixContext, &missDesc,
        1,  // num program groups
        &pgOptions,
        log, &sizeof_log,
        &missProgramGroups[0]
    );

    // Hits
    OptixProgramGroupDesc hitDesc = {};
    hitDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitDesc.raygen.module = module;
    hitDesc.raygen.entryFunctionName = "__hit__radiance";

    optixProgramGroupCreate(
        optixContext, &hitDesc,
        1,  // num program groups
        &pgOptions,
        log, &sizeof_log,
        &hitProgramGroups[0]
    );
}

void OptixRenderer::createPipeline()
{
}

void OptixRenderer::buildSBT()
{
}
