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
        - camera.up * camera.pixelLength.y * ((float)iy - (float)params.windowSize.y * 0.5f)
    );


    optixTrace(params.rootHandle,
        camera.position,
        rayDir,
        0.f,    // tmin
        1e20f,  // tmax
        0.0f,   // rayTime
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,//OPTIX_RAY_FLAG_NONE,
        0,             // SBT offset
        1,               // SBT stride
        0,             // missSBTIndex 
        u0, u1);

    const int r = int(255.99f * pixelColorPRD.x);
    const int g = int(255.99f * pixelColorPRD.y);
    const int b = int(255.99f * pixelColorPRD.z);

    // and write to frame buffer ...
    const uint32_t fbIndex = ix + iy * params.windowSize.x;

    const uint32_t rgba = 0xff000000
        | (r << 0) | (g << 8) | (b << 16);

    params.frame.colorBuffer[fbIndex] = rgba;
}

extern "C" __global__ void __miss__radiance() {
    float3& prd = *(float3*)getPRD<float3>();
    prd = make_float3(1.0f);


}

extern "C" __global__ void __hit__radiance() {
    ChunkData* chunk = (ChunkData*)optixGetSbtDataPointer();

    const int   primID = optixGetPrimitiveIndex();
    const int    vert_idx_offset = primID * 3;

    Vertex v = chunk->verts[chunk->idx[vert_idx_offset]];

    float3& prd = *(float3*)getPRD<float3>();
    prd = v.nor;
}