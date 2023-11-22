#include "shader_commons.h"

/*! launch parameters in constant memory, filled in by optix upon
      optixLaunch (this gets filled in from the buffer we pass to
      optixLaunch) */
extern "C" __constant__ OptixParams params;


static __forceinline__ __device__
void* unpackPointer(uint32_t i0, uint32_t i1)
{
    const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
    void* ptr = reinterpret_cast<void*>(uptr);
    return ptr;
}

static __forceinline__ __device__
void  packPointer(void* ptr, uint32_t& i0, uint32_t& i1)
{
    const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

template<typename T>
static __forceinline__ __device__ T* getPRD()
{
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return reinterpret_cast<T*>(unpackPointer(u0, u1));
}

extern "C" __global__ void __raygen__render() {
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;

    const auto& camera = params.camera;
    float3 pixelColorPRD = make_float3(0.f);

    uint32_t u0, u1;
    packPointer(&pixelColorPRD, u0, u1);

    float3 rayDir = normalize(camera.forward
        - camera.right * camera.pixelLength.x * ((float)ix - (float)params.windowSize.x * 0.5f)
        - camera.up * camera.pixelLength.y * -((float)iy - (float)params.windowSize.y * 0.5f)
    );

    optixTrace(params.rootHandle,
        camera.position,
        rayDir,
        0.f,    // tmin
        1e20f,  // tmax
        0.0f,   // rayTime
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,  //OPTIX_RAY_FLAG_NONE,
        0,  // SBT offset
        1,  // SBT stride
        0,  // missSBTIndex
        u0, u1);

    const int r = int(255.99f * pixelColorPRD.x);
    const int g = int(255.99f * pixelColorPRD.y);
    const int b = int(255.99f * pixelColorPRD.z);

    //const int r = int(255.99f * (ix / (float)(params.windowSize.x)));
    //const int g = int(255.99f * (iy / (float)(params.windowSize.y)));
    //const int b = 0;

    // and write to frame buffer ...
    const uint32_t fbIndex = ix + iy * params.windowSize.x;

    const uint32_t rgba = 0xff000000
        | (r << 0) | (g << 8) | (b << 16);    

    params.frame.colorBuffer[fbIndex] = rgba;
}

extern "C" __global__ void __miss__radiance() {
    float3& prd = *(float3*)getPRD<float3>();
    prd = make_float3(0.5f, 0.8f, 1.0f) * 0.4f;
}

extern "C" __global__ void __closesthit__radiance() {
    const ChunkData& chunkData = *(const ChunkData*)optixGetSbtDataPointer();

    const int primID = optixGetPrimitiveIndex();

    const uint3 vIdx = chunkData.idx[primID];

    const Vertex& v1 = chunkData.verts[vIdx.x];
    const Vertex& v2 = chunkData.verts[vIdx.y];
    const Vertex& v3 = chunkData.verts[vIdx.z];

    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    float2 uv = (1.f - u - v) * v1.uv + u * v2.uv + v * v3.uv;
    float4 diffuseCol = tex2D<float4>(chunkData.tex_diffuse, uv.x, uv.y);

    float3& prd = *(float3*)getPRD<float3>();
    prd = make_float3(diffuseCol);
}

extern "C" __global__ void __anyhit__radiance()
{

}

extern "C" __global__ void __exception__all()
{
    // This assumes that the launch dimensions are matching the size of the output buffer.

    const uint3 theLaunchIndex = optixGetLaunchIndex();

    const int theExceptionCode = optixGetExceptionCode();
    printf("Exception %d at (%u, %u)\n", theExceptionCode, theLaunchIndex.x, theLaunchIndex.y);
}