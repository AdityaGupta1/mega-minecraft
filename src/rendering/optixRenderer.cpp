#include "optixRenderer.hpp"
#include "ShaderList.h"
#include "rendering/renderingUtils.hpp"

#include <stb_image.h>

#undef min
#undef max
#include <optix_stack_size.h>
#include <optix_function_table_definition.h>
#include <cuda_gl_interop.h>

constexpr int numRayTypes = 2;

#if USE_D3D11_RENDERER
OptixRenderer::OptixRenderer(D3D11Renderer* renderer, uvec2* windowSize, Terrain* terrain, Player* player)
    : renderer(renderer), windowSize(windowSize), terrain(terrain), player(player), 
      pboResource(*(renderer->getCudaTextureResource())), vao(-1), pbo(-1), tex_pixels(-1)
#else
OptixRenderer::OptixRenderer(GLFWwindow* window, uvec2* windowSize, Terrain* terrain, Player* player) 
    : window(window), windowSize(windowSize), terrain(terrain), player(player), vao(-1), pbo(-1), tex_pixels(-1)
#endif
{
    int numMaxDrawableChunks = Terrain::getMaxNumDrawableChunks();
    for (int i = 0; i < numMaxDrawableChunks; ++i)
    {
        chunkIdsQueue.push(i);
    }

    gasBufferPtrs.resize(numMaxDrawableChunks);
    host_hitGroupRecords.resize(numMaxDrawableChunks * numRayTypes);

#if USE_UPSCALING
    const uint32_t inWidth = windowSize->x >> 1;
    const uint32_t inHeight = windowSize->y >> 1;
#else
    const uint32_t inWidth = windowSize->x;
    const uint32_t inHeight = windowSize->y;
#endif

    launchParamsBuffer.alloc(sizeof(OptixParams));
    launchParams.windowSize = make_int2(inWidth, inHeight);
    pixels.resize(inWidth * inHeight);

    CUDA_CHECK(cudaMalloc((void**)&dev_chunkInstances, numMaxDrawableChunks * sizeof(OptixInstance)));

    size_t imageSizeBytes = inWidth * inHeight * sizeof(float4);
    CUDA_CHECK(cudaMalloc((void**)&dev_renderBuffer, imageSizeBytes));
    CUDA_CHECK(cudaMalloc((void**)&dev_albedoBuffer, imageSizeBytes));
    CUDA_CHECK(cudaMalloc((void**)&dev_normalBuffer, imageSizeBytes));
    CUDA_CHECK(cudaMalloc((void**)&dev_denoisedBuffer, (windowSize->x * windowSize->y) * sizeof(float4)));
//    CUDA_CHECK(cudaMalloc((void**)&dev_flowBuffer, imageSizeBytes));

#if !USE_D3D11_RENDERER
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    fullscreenTri.bufferVBOs();

    initShader();

    initTexture();
#endif

    setZoomed(false);

    sunAxisForward = normalize(vec3(6.0f, -2.0f, 2.0f));
    sunAxisRight = normalize(cross(sunAxisForward, vec3(0, 1, 0)));
    sunAxisUp = normalize(cross(sunAxisRight, sunAxisForward));
    updateTime(0.f);

    createContext();
}

static void context_log_cb(unsigned int level,
    const char* tag,
    const char* message,
    void*) {
    fprintf(stderr, "[%2d][%12s]: %s\n", (int)level, tag, message);
}

void OptixRenderer::createContext()
{
    //cudaFree(0);
    CU_CHECK(cuCtxGetCurrent(&cudaContext));

    OPTIX_CHECK(optixInit());
    OptixDeviceContextOptions options = {};
#if !defined(NDEBUG)
    options.logCallbackLevel = 4;
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#else
    options.logCallbackLevel = 3;
#endif

    OPTIX_CHECK(optixDeviceContextCreate(cudaContext, &options, &optixContext));
    OPTIX_CHECK(optixDeviceContextSetLogCallback(
        optixContext,
        context_log_cb,
        nullptr,
        options.logCallbackLevel
    ));
    createModule();
    createProgramGroups();
    createPipeline();
    createTextures();
    buildSBT(false);
#if USE_DENOISING
    createDenoiser();
#endif
}

void OptixRenderer::loadTexture(const std::string& path, int textureId)
{
    int width, height, channels;
    unsigned char* host_data = stbi_load(path.c_str(), &width, &height, &channels, 0);

    for (int i = 0; i < width * height; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            auto& col = host_data[i * 4 + j];
            col = (unsigned char)(pow(col / 255.f, 2.2f) * 255.f);
        }
    }

    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<uchar4>();
    int pitch = width * channels * sizeof(uint8_t);

    cudaArray_t& pixelArray = texArrays[textureId];
    CUDA_CHECK(cudaMallocArray(
        &pixelArray,
        &channel_desc,
        width, 
        height
    ));

    CUDA_CHECK(cudaMemcpy2DToArray(pixelArray,
        0, // wOffset 
        0, // hOffset
        host_data,
        pitch, 
        pitch, 
        height,
        cudaMemcpyHostToDevice
    ));

    stbi_image_free(host_data);

    cudaResourceDesc res_desc = {};
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = pixelArray;

    cudaTextureDesc tex_desc = {};
    tex_desc.addressMode[0] = cudaAddressModeWrap;
    tex_desc.addressMode[1] = cudaAddressModeWrap;
    tex_desc.filterMode = cudaFilterModePoint;
    tex_desc.readMode = cudaReadModeNormalizedFloat;
    tex_desc.normalizedCoords = 1;
    tex_desc.maxAnisotropy = 1;
    tex_desc.maxMipmapLevelClamp = 99;
    tex_desc.minMipmapLevelClamp = 0;
    tex_desc.mipmapFilterMode = cudaFilterModePoint;
    tex_desc.borderColor[0] = 1.0f;
    tex_desc.sRGB = 0;

    // Create texture object
    cudaTextureObject_t cuda_tex;
    CUDA_CHECK(cudaCreateTextureObject(&cuda_tex, &res_desc, &tex_desc, nullptr));
    texObjects[textureId] = cuda_tex;
}

void OptixRenderer::createTextures()
{
    int numTextures = 3;
    texArrays.resize(numTextures);
    texObjects.resize(numTextures);

    stbi_set_flip_vertically_on_load(true);

    loadTexture("textures/blocks_diffuse.png", 0);
    loadTexture("textures/blocks_emissive.png", 1);
    loadTexture("textures/blocks_normal.png", 2);
}

void OptixRenderer::buildChunkAccel(const Chunk* chunkPtr)
{
    // copy mesh data to device
    CUBuffer dev_vertices;
    CUBuffer dev_indices;

    dev_vertices.initFromVector(chunkPtr->verts);
    dev_indices.initFromVector(chunkPtr->idx);

    OptixTraversableHandle gasHandle = 0;

    // build triangle_input
    CUdeviceptr d_vertices = dev_vertices.dev_ptr();
    CUdeviceptr d_indices  = dev_indices.dev_ptr();

    if (chunkIdsQueue.empty())
    {
        fprintf(stderr, "chunk ids queue is empty\n");
        exit(-1);
    }

    int chunkId = chunkIdsQueue.front();
    chunkIdsQueue.pop();

    chunkIdsMap[chunkPtr] = chunkId;

    const ChunkData chunkData = {
        (Vertex*)d_vertices,
        (uvec3*)d_indices,
        texObjects[0],
        texObjects[1],
        texObjects[2]
    };

    for (int i = 0; i < hitProgramGroups.size(); i++)
    {
        HitGroupRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(hitProgramGroups[i], &rec)); // TODO: store this once instead of recreating for each chunk
        rec.data = chunkData;
        host_hitGroupRecords[chunkId * numRayTypes + i] = rec;
    }

    OptixBuildInput triangleInput = {};
    triangleInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
      
    triangleInput.triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangleInput.triangleArray.vertexStrideInBytes = sizeof(Vertex);
    triangleInput.triangleArray.numVertices         = (int)chunkPtr->verts.size();
    triangleInput.triangleArray.vertexBuffers       = &d_vertices;
    
    triangleInput.triangleArray.indexFormat         = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangleInput.triangleArray.indexStrideInBytes  = sizeof(glm::uvec3);
    triangleInput.triangleArray.numIndexTriplets    = (int)chunkPtr->idx.size() / 3;
    triangleInput.triangleArray.indexBuffer         = d_indices;
    
    uint32_t triangleInputFlags[1] = { 0 };

    triangleInput.triangleArray.flags = triangleInputFlags;
    triangleInput.triangleArray.numSbtRecords = 1;
    triangleInput.triangleArray.sbtIndexOffsetBuffer = 0;
    triangleInput.triangleArray.sbtIndexOffsetSizeInBytes = 0;
    triangleInput.triangleArray.sbtIndexOffsetStrideInBytes = 0;
    
    // optixAccelComputeMemoryUsage
    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes gasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage
    (optixContext,
        &accelOptions,
        &triangleInput,
        1,  // num_build_inputs
        &gasBufferSizes
    ));

    CUBuffer compactedSizeBuffer;
    compactedSizeBuffer.alloc(sizeof(uint64_t));

    OptixAccelEmitDesc emitDesc;
    emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDesc.result = compactedSizeBuffer.dev_ptr();

    CUBuffer outputBuffer;
    outputBuffer.alloc(gasBufferSizes.outputSizeInBytes);

    CUBuffer tempBuffer;
    tempBuffer.alloc(gasBufferSizes.tempSizeInBytes);

    OPTIX_CHECK(optixAccelBuild(optixContext,
        0, // stream
        &accelOptions,
        &triangleInput,
        1, // numBuildInputs
        tempBuffer.dev_ptr(),
        tempBuffer.size(),

        outputBuffer.dev_ptr(),
        outputBuffer.size(),

        &gasHandle,

        &emitDesc,
        1 // numEmittedProperties
    ));

    cudaDeviceSynchronize();

    // don't free dev_vertices and dev_indices since they're used in hit group records (ChunkData)
    
    // optixAccelCompact
    uint64_t compactedSize;
    compactedSizeBuffer.retrieve(&compactedSize, 1);

    CUBuffer gasBuffer;
    gasBuffer.alloc(compactedSize);
    OPTIX_CHECK(optixAccelCompact(optixContext,
        0, // stream
        gasHandle,
        gasBuffer.dev_ptr(),
        gasBuffer.size(),
        &gasHandle));

    // cleanup
    tempBuffer.free();
    outputBuffer.free();
    compactedSizeBuffer.free();

    // to instance
    OptixInstance gasInstance = {};
    float transform[12] = {
        1, 0, 0, chunkPtr->worldBlockPos.x,
        0, 1, 0, 0,
        0, 0, 1, chunkPtr->worldBlockPos.z
    };
    memcpy(gasInstance.transform, transform, sizeof(float) * 12);
    gasInstance.instanceId = chunkId; // this is NOT the SBT geometry-AS index - that would be if one GAS had multiple build inputs
    gasInstance.visibilityMask = 255;
    gasInstance.sbtOffset = chunkId * numRayTypes; // I_offset (TODO: multiply by number of ray types)
    gasInstance.flags = OPTIX_INSTANCE_FLAG_NONE;
    gasInstance.traversableHandle = gasHandle;

    chunkInstances[chunkPtr] = gasInstance;
    gasBufferPtrs[chunkId] = (void*)gasBuffer.dev_ptr();
}

void OptixRenderer::buildRootAccel()
{
    if (chunkInstances.empty())
    {
        return;
    }

    std::vector<OptixInstance> chunkInstancesVector;
    chunkInstancesVector.reserve(chunkInstances.size());
    for (const auto& elem : chunkInstances)
    {
        chunkInstancesVector.push_back(elem.second);
    }

    CUDA_CHECK(cudaMemcpy(dev_chunkInstances, chunkInstancesVector.data(), chunkInstancesVector.size() * sizeof(OptixInstance), cudaMemcpyHostToDevice));

    rootIAS.buildInput = {};
    rootIAS.buildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    rootIAS.buildInput.instanceArray.instances = (CUdeviceptr)dev_chunkInstances;
    rootIAS.buildInput.instanceArray.numInstances = static_cast<unsigned int>(chunkInstancesVector.size());

    rootIAS.buildOptions = {};
    rootIAS.buildOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
    rootIAS.buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    rootIAS.bufferSizes = {};
    OPTIX_CHECK(optixAccelComputeMemoryUsage(optixContext, &rootIAS.buildOptions, &rootIAS.buildInput, 1, &rootIAS.bufferSizes));

    rootIAS.outputBuffer.resize(rootIAS.bufferSizes.outputSizeInBytes);
    rootIAS.tempBuffer.resize(rootIAS.bufferSizes.tempSizeInBytes);

    OPTIX_CHECK(optixAccelBuild(optixContext, 0, &rootIAS.buildOptions, &rootIAS.buildInput, 1,
        rootIAS.tempBuffer.dev_ptr(), rootIAS.bufferSizes.tempSizeInBytes,
        rootIAS.outputBuffer.dev_ptr(), rootIAS.bufferSizes.outputSizeInBytes,
        &launchParams.rootHandle, nullptr, 0));

    buildSBT(true);

#if RESET_CAMERA_ON_BUILD_IAS
    cameraChanged = true;
#endif
}

void OptixRenderer::destroyChunk(const Chunk* chunkPtr)
{
    const int chunkId = chunkIdsMap[chunkPtr];

    chunkInstances.erase(chunkPtr);
    CUDA_CHECK(cudaFree(gasBufferPtrs[chunkId]));

    const auto& hitGroupRecordData = host_hitGroupRecords[chunkId * numRayTypes].data; // free for only one hit group since the rest use the same pointers
    CUDA_CHECK(cudaFree(hitGroupRecordData.verts));
    CUDA_CHECK(cudaFree(hitGroupRecordData.idx));

    chunkIdsMap.erase(chunkPtr);
    chunkIdsQueue.push(chunkId);
}

static constexpr float fovNormal = glm::radians(47.f);
static constexpr float fovZoomed = glm::radians(20.f);

void OptixRenderer::setZoomed(bool zoomed)
{
    if (zoomed == this->isZoomed)
    {
        return;
    }

    this->isZoomed = zoomed;
    tanFovy = tanf(zoomed ? fovZoomed : fovNormal);
    cameraChanged = true;
}

void OptixRenderer::toggleTimePaused()
{
    this->isTimePaused = !this->isTimePaused;
}

void OptixRenderer::addTime(float deltaTime)
{
    updateTime(deltaTime);
}

inline float3 vec3ToFloat3(glm::vec3 v)
{
    return make_float3(v.x, v.y, v.z);
}

void OptixRenderer::setCamera()
{
#if USE_UPSCALING
    const uint32_t inWidth = windowSize->x >> 1;
    const uint32_t inHeight = windowSize->y >> 1;
#else
    const uint32_t inWidth = windowSize->x;
    const uint32_t inHeight = windowSize->y;
#endif
    float yscaled = tanFovy;
    float xscaled = (yscaled * inWidth) / inHeight;
    launchParams.camera.pixelLength = make_float2(2 * xscaled / (float)inWidth, 2 * yscaled / (float)inHeight);

    launchParams.camera.forward = vec3ToFloat3(player->getForward());
    launchParams.camera.up = vec3ToFloat3(player->getUp());
    launchParams.camera.right = vec3ToFloat3(player->getRight());
    launchParams.camera.position = vec3ToFloat3(player->getPos());
    launchParams.windowSize = make_int2(inWidth, inHeight);
    launchParams.frame.frameId = 0;

    cameraChanged = false;
}

std::vector<char> OptixRenderer::readData(std::string const& filename)
{
    std::ifstream inputData(filename, std::ios::binary);

    if (inputData.fail())
    {
        std::cerr << "ERROR: readData() Failed to open file " << filename << '\n';
        return std::vector<char>();
    }

    // Copy the input buffer to a char vector.
    std::vector<char> data(std::istreambuf_iterator<char>(inputData), {});

    if (inputData.fail())
    {
        std::cerr << "ERROR: readData() Failed to read file " << filename << '\n';
        return std::vector<char>();
    }

    return data;
}

void OptixRenderer::createModule()
{
#if !defined(NDEBUG)
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#else
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE; // MODERATE for Nsight Compute
#endif
    moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;

    pipelineCompileOptions.usesMotionBlur = 0;
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipelineCompileOptions.numPayloadValues = 2;
    pipelineCompileOptions.numAttributeValues = 2;
#if !defined(NDEBUG)
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH;
#else
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";

    pipelineLinkOptions.maxTraceDepth = 1;

    modules.resize(shaderFiles.size());
    for (size_t i = 0; i < shaderFiles.size(); i++) {
        std::vector<char> programData = readData(shaderFiles[i]);

        char log[2048];
        size_t sizeof_log = sizeof(log);

        OPTIX_CHECK(optixModuleCreate(optixContext,
            &moduleCompileOptions,
            &pipelineCompileOptions,
            programData.data(),
            programData.size(),
            log, &sizeof_log,
            &modules[i]
        ));
        if (sizeof_log > 1) std::cout << "Module " << i << " Create:" << log << std::endl;      
    }
}

void OptixRenderer::createProgramGroups()
{
    OptixProgramGroupOptions pgOptions = {};

    // change depending on # materials
    raygenProgramGroups.resize(1);
    missProgramGroups.resize(1);
    hitProgramGroups.resize(2);
    exceptionProgramGroups.resize(1);
     
    // Ray Gen
    OptixProgramGroupDesc rayGenDesc = {};
    rayGenDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    rayGenDesc.raygen.module = modules[0];
    rayGenDesc.raygen.entryFunctionName = "__raygen__render";

    char log[2048];
    size_t sizeof_log = sizeof(log);

    OPTIX_CHECK(optixProgramGroupCreate(
        optixContext, &rayGenDesc,
        1,  // num program groups
        &pgOptions,
        log, &sizeof_log,
        &raygenProgramGroups[0]
    ));

    if (sizeof_log > 1) std::cout << "RayGen PG: " << log << std::endl;

    // Miss
    OptixProgramGroupDesc missDesc = {};
    missDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    missDesc.miss.module = modules[0];
    missDesc.miss.entryFunctionName = "__miss__radiance";

    OPTIX_CHECK(optixProgramGroupCreate(
        optixContext, &missDesc,
        1,  // num program groups
        &pgOptions,
        log, &sizeof_log,
        &missProgramGroups[0]
    ));

    if (sizeof_log > 1) std::cout << "Miss PG: " << log << std::endl;

    // Hits
    OptixProgramGroupDesc hitDesc = {};
    hitDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitDesc.hitgroup.moduleCH = modules[0];
    hitDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    hitDesc.hitgroup.moduleAH = modules[0];
    hitDesc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";
    OPTIX_CHECK(optixProgramGroupCreate(
        optixContext,
        &hitDesc,
        1,  // num program groups
        &pgOptions,
        log, &sizeof_log,
        &hitProgramGroups[0]
    ));

    // no closest hit shader
    hitDesc.hitgroup.entryFunctionNameAH = "__anyhit__shadow";
    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
        &hitDesc,
        1,
        &pgOptions,
        log, &sizeof_log,
        &hitProgramGroups[1]
    ));

    if (sizeof_log > 1) std::cout << "Hit PG: " << log << std::endl;

    // Exception
    OptixProgramGroupDesc excpDesc = {};
    excpDesc.kind = OPTIX_PROGRAM_GROUP_KIND_EXCEPTION;
    excpDesc.exception.module = modules[0];
    excpDesc.exception.entryFunctionName = "__exception__all";
    OPTIX_CHECK(optixProgramGroupCreate(
        optixContext,
        &excpDesc,
        1,
        &pgOptions,
        log, &sizeof_log,
        &exceptionProgramGroups[0]
    ));

    if (sizeof_log > 1) std::cout << "Exception PG: " << log << std::endl;
}

void OptixRenderer::createPipeline()
{
    std::vector<OptixProgramGroup> programGroups;
    for (auto p : raygenProgramGroups)
        programGroups.push_back(p);
    for (auto p : missProgramGroups)
        programGroups.push_back(p);
    for (auto p : hitProgramGroups)
        programGroups.push_back(p);
    for (auto p : exceptionProgramGroups)
        programGroups.push_back(p);

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixPipelineCreate(optixContext,
        &pipelineCompileOptions,
        &pipelineLinkOptions,
        programGroups.data(),
        (int)programGroups.size(),
        log, &sizeof_log,
        &pipeline
    ));

    OptixStackSizes stack_sizes = {};
    for (auto p : programGroups) {
        OPTIX_CHECK(optixUtilAccumulateStackSizes(p, &stack_sizes, pipeline));
    }

    uint32_t max_trace_depth = pipelineLinkOptions.maxTraceDepth;
    uint32_t max_cc_depth = 0;
    uint32_t max_dc_depth = 0;
    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK(optixUtilComputeStackSizes(
        &stack_sizes,
        max_trace_depth,
        max_cc_depth,
        max_dc_depth,
        &direct_callable_stack_size_from_traversal,
        &direct_callable_stack_size_from_state,
        &continuation_stack_size
    ));

    const uint32_t max_traversal_depth = 2;
    OPTIX_CHECK(optixPipelineSetStackSize(
        pipeline,
        direct_callable_stack_size_from_traversal,
        direct_callable_stack_size_from_state,
        continuation_stack_size,
        max_traversal_depth
    ));
}

void OptixRenderer::buildSBT(bool onlyHitGroups)
{
    // ------------------------------------------------------------------
    // build hitgroup records
    // ------------------------------------------------------------------
    hitRecordBuffer.initFromVector(host_hitGroupRecords);
    sbt.hitgroupRecordBase = hitRecordBuffer.dev_ptr();
    sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupRecord);
    sbt.hitgroupRecordCount = (int)host_hitGroupRecords.size();

    if (onlyHitGroups)
    {
        return;
    }

    // ------------------------------------------------------------------
    // build raygen records
    // ------------------------------------------------------------------
    std::vector<RayGenRecord> raygenRecords;
    for (int i = 0; i < raygenProgramGroups.size(); i++) {
        RayGenRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(raygenProgramGroups[i], &rec));
        rec.data = nullptr; /* for now ... */
        raygenRecords.push_back(rec);
    }
    raygenRecordBuffer.initFromVector(raygenRecords);
    sbt.raygenRecord = raygenRecordBuffer.dev_ptr();

    // ------------------------------------------------------------------
    // build miss records
    // ------------------------------------------------------------------   
    std::vector<MissRecord> missRecords;
    for (int i = 0; i < missProgramGroups.size(); i++) {
        MissRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(missProgramGroups[i], &rec));
        rec.data = nullptr; /* for now ... */
        missRecords.push_back(rec);
    }
    missRecordBuffer.initFromVector(missRecords);
    sbt.missRecordBase = missRecordBuffer.dev_ptr();
    sbt.missRecordStrideInBytes = sizeof(MissRecord);
    sbt.missRecordCount = (int)missRecords.size();

    // ------------------------------------------------------------------
    // build exception records
    // ------------------------------------------------------------------
    std::vector<ExceptionRecord> exceptionRecords;
    for (int i = 0; i < exceptionProgramGroups.size(); i++) {
        ExceptionRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(exceptionProgramGroups[i], &rec));
        rec.data = nullptr; /* for now ... */
        exceptionRecords.push_back(rec);
    }
    exceptionRecordBuffer.initFromVector(exceptionRecords);
    sbt.exceptionRecord = exceptionRecordBuffer.dev_ptr();
}

void OptixRenderer::createDenoiser()
{
    if (denoiser)
    {
        OPTIX_CHECK(optixDenoiserDestroy(denoiser));
    };

#if USE_UPSCALING
    const uint32_t inWidth = windowSize->x >> 1;
    const uint32_t inHeight = windowSize->y >> 1;
#else
    const uint32_t inWidth = windowSize->x;
    const uint32_t inHeight = windowSize->y;
#endif

    OptixDenoiserOptions denoiserOptions = {};
    denoiserOptions.guideAlbedo = 1;
    denoiserOptions.guideNormal = 0;
    denoiserOptions.denoiseAlpha = (OptixDenoiserAlphaMode)0;
#if USE_UPSCALING
    OPTIX_CHECK(optixDenoiserCreate(optixContext, OPTIX_DENOISER_MODEL_KIND_UPSCALE2X, &denoiserOptions, &denoiser));
#else
    OPTIX_CHECK(optixDenoiserCreate(optixContext, OPTIX_DENOISER_MODEL_KIND_AOV, &denoiserOptions, &denoiser));
#endif

    OptixDenoiserSizes denoiserReturnSizes;
    OPTIX_CHECK(optixDenoiserComputeMemoryResources(denoiser, windowSize->x, windowSize->y, &denoiserReturnSizes));

    denoiserScratch.resize(denoiserReturnSizes.withoutOverlapScratchSizeInBytes);
    denoiserState.resize(denoiserReturnSizes.stateSizeInBytes);
    denoiserAvgCol.resize(3 * sizeof(float));

    denoiserLayer.input = createOptixImage2D(inWidth, inHeight, (CUdeviceptr)dev_renderBuffer);
    denoiserLayer.output = createOptixImage2D(windowSize->x, windowSize->y, (CUdeviceptr)dev_denoisedBuffer);

#if 0 // abandoning temporal for now
    denoiserLayer.previousOutput = createOptixImage2D(windowSize->x, windowSize->y);

    CUDA_CHECK(cudaMemset(dev_flowBuffer, 0, inWidth * inHeight * sizeof(float4)));
    denoiserGuideLayer.flow = createOptixImage2D(inWidth, inHeight);

    void* internalMemIn = 0;
    void* internalMemOut = 0;
    size_t internalSize = windowSize->x * windowSize->y * denoiserReturnSizes.internalGuideLayerPixelSizeInBytes;
    CUDA_CHECK(cudaMalloc(&internalMemIn, internalSize));
    CUDA_CHECK(cudaMalloc(&internalMemOut, internalSize));
    CUDA_CHECK(cudaMemset(internalMemIn, 0, internalSize));

    denoiserGuideLayer.previousOutputInternalGuideLayer.data = (CUdeviceptr)internalMemIn;
    denoiserGuideLayer.previousOutputInternalGuideLayer.width = windowSize->x;
    denoiserGuideLayer.previousOutputInternalGuideLayer.height = windowSize->y;
    denoiserGuideLayer.previousOutputInternalGuideLayer.pixelStrideInBytes = unsigned(denoiserReturnSizes.internalGuideLayerPixelSizeInBytes);
    denoiserGuideLayer.previousOutputInternalGuideLayer.rowStrideInBytes = denoiserGuideLayer.previousOutputInternalGuideLayer.width * denoiserGuideLayer.previousOutputInternalGuideLayer.pixelStrideInBytes;
    denoiserGuideLayer.previousOutputInternalGuideLayer.format = OPTIX_PIXEL_FORMAT_INTERNAL_GUIDE_LAYER;

    denoiserGuideLayer.outputInternalGuideLayer = denoiserGuideLayer.previousOutputInternalGuideLayer;
    denoiserGuideLayer.outputInternalGuideLayer.data = (CUdeviceptr)internalMemOut;

    // flowTrustworthiness?
#endif
    denoiserGuideLayer.albedo = createOptixImage2D(inWidth, inHeight, (CUdeviceptr)dev_albedoBuffer);
    denoiserGuideLayer.normal = createOptixImage2D(inWidth, inHeight, (CUdeviceptr)dev_normalBuffer);

    OPTIX_CHECK(optixDenoiserSetup(
        denoiser,
        0, // stream
        inWidth, inHeight,
        denoiserState.dev_ptr(),
        denoiserState.size(),
        denoiserScratch.dev_ptr(),
        denoiserScratch.size()
    ));

    denoiserParams.hdrAverageColor = denoiserAvgCol.dev_ptr();
    denoiserParams.blendFactor = 0.f;
    denoiserParams.temporalModeUsePreviousLayers = 0;
}

void OptixRenderer::render(float deltaTime)
{
    if (!isTimePaused)
    {
        updateTime(deltaTime);
    }

    if (cameraChanged)
    {
        setCamera();
    }

    optixRenderFrame();
    updateFrame();
}

void OptixRenderer::onResize()
{
#if USE_UPSCALING
    const uint32_t inWidth = windowSize->x >> 1;
    const uint32_t inHeight = windowSize->y >> 1;
#else
    const uint32_t inWidth = windowSize->x;
    const uint32_t inHeight = windowSize->y;
#endif

    launchParams.windowSize = make_int2(inWidth, inHeight);
    pixels.resize(inWidth * inHeight);

    CUDA_CHECK(cudaFree(dev_renderBuffer));
    CUDA_CHECK(cudaFree(dev_albedoBuffer));
    CUDA_CHECK(cudaFree(dev_normalBuffer));
    CUDA_CHECK(cudaFree(dev_denoisedBuffer));

    size_t imageSizeBytes = inWidth * inHeight * sizeof(float4);
    CUDA_CHECK(cudaMalloc((void**)&dev_renderBuffer, imageSizeBytes));
    CUDA_CHECK(cudaMalloc((void**)&dev_albedoBuffer, imageSizeBytes));
    CUDA_CHECK(cudaMalloc((void**)&dev_normalBuffer, imageSizeBytes));
    CUDA_CHECK(cudaMalloc((void**)&dev_denoisedBuffer, (windowSize->x * windowSize->y) * sizeof(float4)));

    pboResource = *(renderer->getCudaTextureResource());
}

void OptixRenderer::updateTime(float deltaTime)
{
    time += deltaTime;
    launchParams.time = time;

    sunTime += deltaTime * -0.025f;
    const glm::vec3 rotatedAxisRight = cosf(sunTime) * sunAxisRight + sinf(sunTime) * sunAxisUp;
    const glm::vec3 rotatedAxisUp = normalize(cross(rotatedAxisRight, sunAxisForward));

    mat3 sunRotateMat = glm::mat3(rotatedAxisRight, sunAxisForward, rotatedAxisUp);

    launchParams.sunDir = vec3ToFloat3(glm::normalize(sunRotateMat * glm::vec3(1.f, 0.15f, 0.f)));
    launchParams.moonDir = vec3ToFloat3(glm::normalize(sunRotateMat * glm::vec3(-1.f, 0.50f, 0.f)));

    // store rows of sunRotateMat as columns of starsRotateMat to do transpose (for inverting sunRotateMat)
    launchParams.starsRotateMatX = make_float3(sunRotateMat[0][0], sunRotateMat[1][0], sunRotateMat[2][0]);
    launchParams.starsRotateMatY = make_float3(sunRotateMat[0][1], sunRotateMat[1][1], sunRotateMat[2][1]);
    launchParams.starsRotateMatZ = make_float3(sunRotateMat[0][2], sunRotateMat[1][2], sunRotateMat[2][2]);

    cameraChanged = true;
}

void OptixRenderer::optixRenderFrame()
{
    if (launchParams.windowSize.x == 0/* || chunkInstances.empty()*/)
    {
        return;
    }

    launchParams.frame.colorBuffer = dev_renderBuffer;
    launchParams.frame.albedoBuffer = dev_albedoBuffer;
    launchParams.frame.normalBuffer = dev_normalBuffer;
    launchParamsBuffer.populate(&launchParams, 1);
    launchParams.frame.frameId++;

    OPTIX_CHECK(optixLaunch(/*! pipeline we're launching launch: */
        pipeline,
        0, // stream

        /*! parameters and SBT */
        launchParamsBuffer.dev_ptr(),
        launchParamsBuffer.size(),
        &sbt,

        /*! dimensions of the launch: */
        launchParams.windowSize.x,
        launchParams.windowSize.y,
        1
    ));

    size_t pboSize;
    CUDA_CHECK(cudaGraphicsMapResources(1, &pboResource, 0));
#if USE_D3D11_RENDERER
    CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&pboArray, pboResource, 0, 0));
    // cudaMemcpy2DFromArray(dev_denoisedBuffer, windowSize->x * sizeof(float4), pboArray, 0, 0, windowSize->x * sizeof(float4), windowSize->y, cudaMemcpyDeviceToDevice);
#else
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&dev_denoisedBuffer, &pboSize, pboResource));
#endif

#if USE_DENOISING
#if 0 // abandoning temporal
    OptixImage2D temp = denoiserGuideLayer.previousOutputInternalGuideLayer;
    denoiserGuideLayer.previousOutputInternalGuideLayer = denoiserGuideLayer.outputInternalGuideLayer;
    denoiserGuideLayer.outputInternalGuideLayer = temp;

    temp = denoiserLayer.previousOutput;
    denoiserLayer.previousOutput = denoiserLayer.output;
    denoiserLayer.output = temp;

    denoiserParams.temporalModeUsePreviousLayers = 1;
#endif
    OPTIX_CHECK(optixDenoiserComputeAverageColor(
        denoiser,
        0, // stream
        &denoiserLayer.input,
        denoiserAvgCol.dev_ptr(),
        denoiserScratch.dev_ptr(),
        denoiserScratch.size()
    ));

    OPTIX_CHECK(optixDenoiserInvoke(denoiser,
        0, // stream
        &denoiserParams,
        denoiserState.dev_ptr(),
        denoiserState.size(),
        &denoiserGuideLayer,
        &denoiserLayer,
        1, // numLayers
        0, // inputOffsetX
        0, // inputOffsetY
        denoiserScratch.dev_ptr(),
        denoiserScratch.size()
    ));
#else
    cudaMemcpy(dev_denoisedBuffer, dev_renderBuffer, windowSize->x * windowSize->y * sizeof(float4), cudaMemcpyDeviceToDevice);
#endif

#if USE_D3D11_RENDERER
    CUDA_CHECK(cudaMemcpy2DToArray(pboArray, 0, 0, (void*)dev_denoisedBuffer, windowSize->x * sizeof(float4), windowSize->x * sizeof(float4), windowSize->y, cudaMemcpyDeviceToDevice));
#endif
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &pboResource));
}

#if USE_D3D11_RENDERER
void OptixRenderer::updateFrame()
{
    renderer->Draw();
}
#else
void OptixRenderer::initShader()
{
    bool success = true;
    success &= postprocessingShader.create("shaders/passthrough_uvs.vert.glsl", "shaders/postprocess_tone_mapping.frag.glsl");

    if (RenderingUtils::printGLErrors())
    {
        std::cerr << "ERROR: Renderer::initShaders()" << std::endl;
    }
}

void OptixRenderer::initTexture()
{
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, windowSize->x * windowSize->y * sizeof(glm::vec4), NULL, GL_DYNAMIC_COPY);
    uint32_t devCnt;
    int devNum;
    CUDA_CHECK(cudaGLGetDevices(&devCnt, &devNum, 2, cudaGLDeviceListAll));
    fprintf(stderr, "GL using %u devices. Current frame is using GPU %d\n", devCnt, devNum);
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&pboResource, pbo, cudaGraphicsRegisterFlagsWriteDiscard));

    glActiveTexture(GL_TEXTURE0);
    glGenTextures(1, &tex_pixels);
    glBindTexture(GL_TEXTURE_2D, tex_pixels);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, windowSize->x, windowSize->y, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    postprocessingShader.setTexBufColor(0);
}


void OptixRenderer::updateFrame()
{
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_pixels);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, windowSize->x, windowSize->y, GL_RGBA, GL_FLOAT, NULL);

    postprocessingShader.draw(fullscreenTri);

    glfwSwapBuffers(window);
}
#endif