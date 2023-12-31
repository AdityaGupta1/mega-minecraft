#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda.h>
#include <optix.h>
#include <optix_stubs.h>

#include <vector>
#include <iostream>
#include <string>
#include <fstream>

#include "cuda/cudaUtils.hpp"
#include "player/player.hpp"
#include "terrain/terrain.hpp"
#include "util/common.h"
#include "shaderProgram.hpp"
#include "fullscreenTri.hpp"

#if USE_D3D11_RENDERER
#include "d3d11Renderer.h"
#endif

class Terrain;
class Chunk;

template<class T>
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) Record
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef Record<void*>      RayGenRecord;
typedef Record<ChunkData>  HitGroupRecord;
typedef Record<void*>      MissRecord;
typedef Record<void*>      ExceptionRecord;

class OptixRenderer
{
public:
#if USE_D3D11_RENDERER
    OptixRenderer(D3D11Renderer* renderer, uvec2* windowSize, Terrain* terrain, Player* player);
#else
    OptixRenderer(GLFWwindow* window, uvec2* windowSize, Terrain* terrain, Player* player);
#endif

    void render(float deltaTime);
    void onResize();

protected:
#if USE_D3D11_RENDERER
    D3D11Renderer* renderer{ nullptr };
#else
    GLFWwindow* window{ nullptr };
#endif
    uvec2* windowSize{ nullptr };
    Terrain* terrain{ nullptr };
    Player* player{ nullptr };
    OptixParams launchParams = {};
    CUBuffer launchParamsBuffer;

    bool cameraChanged{ true };
    bool isZoomed{ true };
    float tanFovy;

    CUcontext          cudaContext = {};

    OptixDeviceContext optixContext = {};

    OptixPipeline               pipeline;
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    OptixPipelineLinkOptions    pipelineLinkOptions = {};

    std::vector<OptixModule>    modules;
    OptixModuleCompileOptions   moduleCompileOptions = {};

    std::vector<cudaArray_t> texArrays;
    std::vector<cudaTextureObject_t> texObjects;

    struct {
        CUBuffer outputBuffer;
        CUBuffer tempBuffer;
        OptixBuildInput buildInput;
        OptixAccelBuildOptions buildOptions;
        OptixAccelBufferSizes bufferSizes;
    } rootIAS;

    std::queue<int> chunkIdsQueue;

    OptixInstance* dev_chunkInstances;
    std::unordered_map<const Chunk*, OptixInstance> chunkInstances;
    std::vector<void*> gasBufferPtrs;
    
    std::unordered_map<const Chunk*, int> chunkIdsMap;
    std::vector<HitGroupRecord> host_hitGroupRecords;

    std::vector<OptixProgramGroup> raygenProgramGroups;
    std::vector<OptixProgramGroup> missProgramGroups;
    std::vector<OptixProgramGroup> hitProgramGroups;
    std::vector<OptixProgramGroup> exceptionProgramGroups;

    CUBuffer raygenRecordBuffer;
    CUBuffer missRecordBuffer;
    CUBuffer hitRecordBuffer;
    CUBuffer exceptionRecordBuffer;
    OptixShaderBindingTable sbt = {};

    CUBuffer playerInfoBuffer;
    cudaGraphicsResource_t pboResource;
#if USE_D3D11_RENDERER
    cudaArray_t pboArray;
#endif

    OptixDenoiser denoiser{ nullptr };
    OptixDenoiserParams denoiserParams = {};
    OptixDenoiserGuideLayer denoiserGuideLayer = {};
    OptixDenoiserLayer denoiserLayer = {};
    CUBuffer denoiserScratch;
    CUBuffer denoiserState;
    CUBuffer denoiserAvgCol;

    float4* dev_renderBuffer;
    float4* dev_albedoBuffer;
    float4* dev_normalBuffer;
    float4* dev_denoisedBuffer;
//    float4* dev_flowBuffer;

    bool isTimePaused{ true };
    float time{ 0 };
    float sunTime{ 0.4f };

    glm::vec3 sunAxisForward{};
    glm::vec3 sunAxisRight{};
    glm::vec3 sunAxisUp{};

    void createContext();

    void loadTexture(const std::string& path, int textureId);
    void createTextures();

public:
    void buildChunkAccel(const Chunk* chunkPtr);
    void buildRootAccel();

    void destroyChunk(const Chunk* chunkPtr);

    void setZoomed(bool zoomed);
    void toggleTimePaused();
    void addTime(float deltaTime);

    void setCamera();

protected:
    std::vector<char> readData(std::string const& filename);
    void createModule();
    void createProgramGroups();
    void createPipeline();
    void buildSBT(bool onlyHitGroups);
    void createDenoiser();

    void updateTime(float deltaTime);
    void optixRenderFrame();
    void updateFrame();

    static OptixImage2D createOptixImage2D(unsigned int width, unsigned int height, CUdeviceptr mem = 0)
    {
        OptixImage2D oi;
        if (mem) {
            oi.data = mem;
        } 
        else {
            CUDA_CHECK(cudaMalloc((void**)(&oi.data), width * height * sizeof(float4)));
        }
        oi.width = width;
        oi.height = height;
        oi.rowStrideInBytes = width * sizeof(float4);
        oi.pixelStrideInBytes = sizeof(float4);
        oi.format = OPTIX_PIXEL_FORMAT_FLOAT4;
        return oi;
    }

    static void copyOptixImage2D(OptixImage2D& dest, const OptixImage2D& src)
    {
        CUDA_CHECK(cudaMemcpy((void*)dest.data, (void*)src.data, src.width * src.height * sizeof(float4), cudaMemcpyDeviceToDevice));
    }

    // GL stuff

    ShaderProgram postprocessingShader;

    std::vector<glm::vec4> pixels;

    FullscreenTri fullscreenTri;

    GLuint vao;

    GLuint pbo;
    GLuint tex_pixels;

#if !USE_D3D11_RENDERER
    void initShader();
    void initTexture();
#endif
};
