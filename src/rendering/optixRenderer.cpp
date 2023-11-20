#include "optixRenderer.hpp"
#include "ShaderList.h"
#include "rendering/renderingUtils.hpp"

#include <stb_image.h>

#undef min
#undef max
#include <optix_stack_size.h>
#include <optix_function_table_definition.h>

template <typename T>
struct Record
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef Record<void*>   RayGenRecord;
typedef Record<void*>     MissRecord;
typedef Record<ChunkData> HitGroupRecord;

OptixRenderer::OptixRenderer(GLFWwindow* window, ivec2* windowSize, Terrain* terrain, Player* player) 
    : window(window), windowSize(windowSize), terrain(terrain), player(player), vao(-1), tex_pixels(-1)
{
    createContext();

    const float fovy = 26.f;
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * windowSize->x) / windowSize->y;
    launchParams.camera.pixelLength = make_float2(2 * xscaled / (float)windowSize->x, 2 * yscaled / (float)windowSize->y);
    setCamera();

    launchParamsBuffer.alloc(sizeof(OptixParams));
    frameBuffer.alloc(windowSize->x * windowSize->y * sizeof(uint32_t));
    launchParams.windowSize = make_int2(windowSize->x, windowSize->y);
    pixels.resize(windowSize->x * windowSize->y);

    cudaStreamCreate(&stream);

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    fullscreenTri.bufferVBOs();

    initShader();

    initTexture();
}
static void context_log_cb(unsigned int level,
    const char* tag,
    const char* message,
    void*) {
    fprintf(stderr, "[%2d][%12s]: %s\n", (int)level, tag, message);
}

void OptixRenderer::createContext()
{
    cuCtxCreate(&cudaContext, 0, 0);

    //cudaFree(0);
    OPTIX_CHECK(optixInit());
    OptixDeviceContextOptions options = {};
    options.logCallbackLevel = 4;
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;

    OPTIX_CHECK(optixDeviceContextCreate(cudaContext, &options, &optixContext));
    OPTIX_CHECK(optixDeviceContextSetLogCallback
    (optixContext, context_log_cb, nullptr, 4));
    createModule();
    createProgramGroups();
    createPipeline();
    createTextures();
    buildSBT();
}

void OptixRenderer::createTextures()
{
    // in case we ever get more textures for who knows what
    int numTextures = 1;

    texArrays.resize(numTextures);
    texObjects.resize(numTextures);

    stbi_set_flip_vertically_on_load(true);

    int width, height, channels;
    unsigned char* data = stbi_load("textures/blocks_diffuse.png", &width, &height, &channels, 0);

    for (int i = 0; i < width * height; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            auto& col = data[i * 4 + j];
            col = (unsigned char)(pow(col / 255.f, 2.2f) * 255.f);
        }
    }
    Texture t = {};
    t.host_buffer = data;
    t.width = width;
    t.height = height;
    t.channels = channels;
    textures.push_back(t);

    for (int textureID = 0; textureID < numTextures; textureID++) {
        auto texture = textures[textureID];

        cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<uchar4>();
        int32_t width = texture.width;
        int32_t height = texture.height;
        int32_t channels = texture.channels;
        int32_t pitch = width * channels * sizeof(uint8_t);

        cudaArray_t& pixelArray = texArrays[textureID];
        CUDA_CHECK(cudaMallocArray(&pixelArray,
            &channel_desc,
            width, height));

        CUDA_CHECK(cudaMemcpy2DToArray(pixelArray,
            /* offset */0, 0,
            texture.host_buffer,
            pitch, pitch, height,
            cudaMemcpyHostToDevice));

        cudaResourceDesc res_desc = {};
        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = pixelArray;

        cudaTextureDesc tex_desc = {};
        tex_desc.addressMode[0] = cudaAddressModeWrap;
        tex_desc.addressMode[1] = cudaAddressModeWrap;
        tex_desc.filterMode = cudaFilterModeLinear;
        tex_desc.readMode = cudaReadModeNormalizedFloat;
        tex_desc.normalizedCoords = 1;
        tex_desc.maxAnisotropy = 1;
        tex_desc.maxMipmapLevelClamp = 99;
        tex_desc.minMipmapLevelClamp = 0;
        tex_desc.mipmapFilterMode = cudaFilterModePoint;
        tex_desc.borderColor[0] = 1.0f;
        tex_desc.sRGB = 0;

        // Create texture object
        cudaTextureObject_t cuda_tex = 0;
        CUDA_CHECK(cudaCreateTextureObject(&cuda_tex, &res_desc, &tex_desc, nullptr));
        texObjects[textureID] = cuda_tex;
    }
}

void OptixRenderer::buildRootAccel()
{
    chunkInstancesBuffer.initFromVector(chunkInstances);

    printf("building root accel: %d\n", chunkInstances.size());
    
    rootIAS.buildInput = {};
    rootIAS.buildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    rootIAS.buildInput.instanceArray.instances = chunkInstancesBuffer.dev_ptr();
    rootIAS.buildInput.instanceArray.numInstances = static_cast<unsigned int>(chunkInstances.size());

    rootIAS.buildOptions = {};
    rootIAS.buildOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    rootIAS.buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    rootIAS.bufferSizes = {};
    OPTIX_CHECK(optixAccelComputeMemoryUsage(optixContext, &rootIAS.buildOptions, &rootIAS.buildInput, 1, &rootIAS.bufferSizes));

    rootIAS.outputBuffer.alloc(rootIAS.bufferSizes.outputSizeInBytes);
    rootIAS.tempBuffer.alloc(rootIAS.bufferSizes.tempSizeInBytes);
    
    OPTIX_CHECK(optixAccelBuild(optixContext, 0, &rootIAS.buildOptions, &rootIAS.buildInput, 1, 
        rootIAS.tempBuffer.dev_ptr(), rootIAS.bufferSizes.tempSizeInBytes, 
        rootIAS.outputBuffer.dev_ptr(), rootIAS.bufferSizes.outputSizeInBytes, 
        &launchParams.rootHandle, nullptr, 0));
}

void OptixRenderer::buildChunkAccel(const Chunk* c)
{
    // copy mesh data to device
    CUBuffer dev_vertices;
    CUBuffer dev_indices;

    dev_vertices.initFromVector(c->verts);
    dev_indices.initFromVector(c->idx);

    vertexBuffer.push_back(dev_vertices);
    indexBuffer.push_back(dev_indices);

    OptixTraversableHandle gasHandle = 0;

    // build triangle_input
    CUdeviceptr d_vertices = dev_vertices.dev_ptr();
    CUdeviceptr d_indices  = dev_indices.dev_ptr();

    OptixBuildInput triangleInput = {};
    triangleInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
      
    triangleInput.triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangleInput.triangleArray.vertexStrideInBytes = sizeof(Vertex);
    triangleInput.triangleArray.numVertices         = (int)c->verts.size();
    triangleInput.triangleArray.vertexBuffers       = &d_vertices;
    
    triangleInput.triangleArray.indexFormat         = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangleInput.triangleArray.indexStrideInBytes  = sizeof(glm::uvec3);
    triangleInput.triangleArray.numIndexTriplets    = (int)c->idx.size() / 3;
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

    CUBuffer tempBuffer;
    tempBuffer.alloc(gasBufferSizes.tempSizeInBytes);

    CUBuffer outputBuffer;
    outputBuffer.alloc(gasBufferSizes.outputSizeInBytes);

    OPTIX_CHECK(optixAccelBuild(optixContext,
        0/*stream*/,
        &accelOptions,
        &triangleInput,
        1,
        tempBuffer.dev_ptr(),
        tempBuffer.size(),

        outputBuffer.dev_ptr(),
        outputBuffer.size(),

        &gasHandle,

        &emitDesc, 1
    ));

    cudaDeviceSynchronize();

    // I don't think you can free the device vertices and indices, cause GAS needs to access this
    //dev_vertices.free();
    //dev_indices.free();
    
    // optixAccelCompact
    uint64_t compactedSize;
    compactedSizeBuffer.retrieve(&compactedSize, 1);

    CUBuffer gasBuffer;

    gasBuffer.alloc(compactedSize);
    OPTIX_CHECK(optixAccelCompact(optixContext,
        0/*stream*/,
        gasHandle,
        gasBuffer.dev_ptr(),
        gasBuffer.size(),
        &gasHandle));

    // cleanup
    //outputBuffer.free();
    //tempBuffer.free();
    //compactedSizeBuffer.free();

    const unsigned int id = chunkInstances.size();

    // to instance
    OptixInstance gasInstance = {};
    float transform[12] = { 1,0,0, c->worldBlockPos.x,0,1,0,0,0,0,1,c->worldBlockPos.z };
    memcpy(gasInstance.transform, transform, sizeof(float) * 12);
    gasInstance.instanceId = id;
    gasInstance.visibilityMask = 255;
    gasInstance.sbtOffset = id;
    gasInstance.flags = OPTIX_INSTANCE_FLAG_NONE;
    gasInstance.traversableHandle = gasHandle;

    chunkInstances.push_back(gasInstance);
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
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;

    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    pipelineCompileOptions.numPayloadValues = 2;
    pipelineCompileOptions.numAttributeValues = 2;
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";

    pipelineLinkOptions.maxTraceDepth = 2;

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
    hitProgramGroups.resize(1);
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

    uint32_t max_trace_depth = 2;
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

    const uint32_t max_traversal_depth = 1;
    OPTIX_CHECK(optixPipelineSetStackSize(
        pipeline,
        direct_callable_stack_size_from_traversal,
        direct_callable_stack_size_from_state,
        continuation_stack_size,
        max_traversal_depth
    ));
}

void OptixRenderer::buildSBT()
{
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
    // build hitgroup records
    // ------------------------------------------------------------------
    std::vector<HitGroupRecord> hitgroupRecords;
    int numChunks = chunkInstances.size();
    for (int c = 0; c < numChunks; c++) {
        for (int i = 0; i < hitProgramGroups.size(); i++) {
            HitGroupRecord rec;
            OPTIX_CHECK(optixSbtRecordPackHeader(hitProgramGroups[i], &rec));
            rec.data.verts = (Vertex*)vertexBuffer[c].dev_ptr();
            rec.data.texture = texObjects[0];
            rec.data.idx = (GLuint*)indexBuffer[c].dev_ptr();
            hitgroupRecords.push_back(rec);
        }
    }
    hitRecordBuffer.initFromVector(hitgroupRecords);
    sbt.hitgroupRecordBase = hitRecordBuffer.dev_ptr();
    sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupRecord);
    sbt.hitgroupRecordCount = (int)hitgroupRecords.size();

    // ------------------------------------------------------------------
    // build exception records
    // ------------------------------------------------------------------
    std::vector<Record<void*>> exceptionRecords;
    for (int i = 0; i < exceptionProgramGroups.size(); i++) {
        Record<void*> rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(exceptionProgramGroups[i], &rec));
        rec.data = nullptr; /* for now ... */
        exceptionRecords.push_back(rec);
    }
    exceptionRecordBuffer.initFromVector(exceptionRecords);
    sbt.exceptionRecord = missRecordBuffer.dev_ptr();
}

void OptixRenderer::optixRenderFrame()
{
    if (launchParams.windowSize.x == 0) return;

    setCamera();

    //for (const Chunk* c : terrain->getDrawableChunks()) {
    //   buildChunkAccel(c);
    //}
    //buildRootAccel();

    launchParams.frame.colorBuffer = (uint32_t*) frameBuffer.dev_ptr();
    launchParamsBuffer.populate(&launchParams, 1);
    launchParams.frame.frameId++;

    OPTIX_CHECK(optixLaunch(/*! pipeline we're launching launch: */
        pipeline,
        stream,
        /*! parameters and SBT */
        launchParamsBuffer.dev_ptr(),
        launchParamsBuffer.size(),
        &sbt,
        /*! dimensions of the launch: */
        launchParams.windowSize.x,
        launchParams.windowSize.y,
        1
    ));

    //printf("framebuffer byte size: %u\n", frameBuffer.byteSize);
    //printf("window x * y: %d\n", launchParams.windowSize.x * launchParams.windowSize.y);

    frameBuffer.retrieve(pixels.data(), windowSize->x * windowSize->y);
}

inline float3 vec3ToFloat3(glm::vec3 v)
{
    return make_float3(v.x, v.y, v.z);
}

void OptixRenderer::setCamera()
{
    launchParams.camera.forward = vec3ToFloat3(player->getForward());
    launchParams.camera.up = vec3ToFloat3(player->getUp());
    launchParams.camera.right = vec3ToFloat3(player->getRight());
    launchParams.camera.position = vec3ToFloat3(player->getPos());
    launchParams.windowSize = make_int2(windowSize->x, windowSize->y);
}

void OptixRenderer::initShader()
{
    bool success = true;
    success &= passthroughUvsShader.create("shaders/passthrough_uvs.vert.glsl", "shaders/passthrough_uvs.frag.glsl");

    if (RenderingUtils::printGLErrors())
    {
        std::cerr << "ERROR: Renderer::initShaders()" << std::endl;
    }
}

void OptixRenderer::initTexture()
{
    glGenTextures(1, &tex_pixels);
    glBindTexture(GL_TEXTURE_2D, tex_pixels);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, windowSize->x, windowSize->y, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());

    passthroughUvsShader.setTexBufColor(0);
}


void OptixRenderer::updateFrame()
{
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, windowSize->x, windowSize->y, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
    passthroughUvsShader.draw(fullscreenTri);
    glfwSwapBuffers(window);
}
