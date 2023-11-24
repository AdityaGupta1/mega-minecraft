#include <optix.h>
#include "shader_commons.h"
#include "random_number_generators.h"

#define PI                3.14159265358979323846264338327f
#define TWO_PI            6.28318530717958647692528676655f
#define PI_OVER_TWO       1.57079632679489661923132169163f
#define PI_OVER_FOUR      0.78539816339744830961566084581f
#define SQRT_2            1.41421356237309504880168872420f
#define SQRT_ONE_THIRD    0.57735026918962576450914878050f
#define INV_PI            0.31830988618379067153776752674f

#define NUM_SAMPLES 4
#define MAX_RAY_DEPTH 2

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

static __forceinline__ __device__ 
float powerHeuristics(int nf, float pdf_f, int ng, float pdf_g) {
    float f = nf * pdf_f;
    float g = ng * pdf_g;
    return f * f / (f * f + g * g);
}

__device__ float3 calculateDirectionNotNormal(const float3 normal)
{
    if (fabs(normal.x) < SQRT_ONE_THIRD)
    {
        return make_float3(1, 0, 0);
    }
    else if (fabs(normal.y) < SQRT_ONE_THIRD)
    {
        return make_float3(0, 1, 0);
    }
    else
    {
        return make_float3(0, 0, 1);
    }
}

__device__ float3 calculateRandomDirectionInHemisphere(float3 normal, float2 sample)
{
    const float up = sqrt(sample.x); // cos(theta)
    const float over = sqrt(1.f - sample.x); // sin(theta)
    const float around = sample.y * TWO_PI;

    // Use not-normal direction to generate two perpendicular directions
    const float3 perpendicularDirection1 = normalize(cross(normal, calculateDirectionNotNormal(normal)));
    const float3 perpendicularDirection2 = normalize(cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__device__ float3 sampleSun(float2 sample)
{
    // find radius and theta in sun space

    // Use not-normal direction to generate two perpendicular directions
    const float3 normal = params.sunDir;

    const float3 perpendicularDirection1 = normalize(cross(normal, calculateDirectionNotNormal(normal)));
    const float3 perpendicularDirection2 = normalize(cross(normal, perpendicularDirection1));

    const float around = sample.y * TWO_PI; // theta

    float3 dir = cos(around) * perpendicularDirection1 + sin(around) * perpendicularDirection2;

    float3 normal2 = normal * normal;
    float max_r = (0.99f - normal2.x - normal2.y - normal2.z) / dot(dir, normal);

    // TODO: so I cheesed this a bit for now but 0.01 should be max_r

    return normal + sample.x * 0.01 * dir;
}

extern "C" __global__ void __raygen__render() {
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;

    const int dx = optixGetLaunchDimensions().x;

    PRD prd;
    prd.seed = tea<4>(iy * dx + ix, params.frame.frameId);

    const auto& camera = params.camera;
    float2 squareSample = rng2(prd.seed);
    float3 rayDir = normalize(camera.forward
        - camera.right * camera.pixelLength.x * ((float)ix - (float)params.windowSize.x * 0.5f + squareSample.x)
        - camera.up * camera.pixelLength.y * -((float)iy - (float)params.windowSize.y * 0.5f + squareSample.y)
    );

    uint32_t u0, u1;
    packPointer(&prd, u0, u1);

    float3 finalColor = make_float3(0);
    float3 finalAlbedo = make_float3(0);
    float3 finalNormal = make_float3(0);

    for (int sample = 0; sample < NUM_SAMPLES; ++sample)
    {
        prd.isDone = false;
        prd.foundLightSource = false;
        prd.rayColor = make_float3(1.f);
        prd.pixelColor = make_float3(0.f);
        prd.isect.pos = camera.position;
        prd.isect.newDir = rayDir;

        for (int depth = 0; depth < MAX_RAY_DEPTH && !prd.isDone; ++depth)
        {
            // 1. BSDF

            optixTrace(params.rootHandle,
                prd.isect.pos,
                prd.isect.newDir,
                0.f,    // tmin
                1e20f,  // tmax
                0.0f,   // rayTime
                OptixVisibilityMask(255),
                OPTIX_RAY_FLAG_NONE,  // OPTIX_RAY_FLAG_NONE,
                0,  // SBT offset
                1,  // SBT stride
                0,  // missSBTIndex
                u0, u1);

            if (!prd.isDone) {
                // MIS: sample light source
                // 2. pdf from Sun & random point on sun
                float2 xi = rng2(prd.seed);

                float3 random_d = sampleSun(xi);

                // 3. test sun intersection
                prd.foundLightSource = true;

                optixTrace(params.rootHandle,
                    prd.isect.pos,
                    random_d,
                    0.f,    // tmin
                    1e20f,  // tmax
                    0.0f,   // rayTime
                    OptixVisibilityMask(255),
                    OPTIX_RAY_FLAG_NONE,  // OPTIX_RAY_FLAG_NONE,
                    1,  // SBT offset
                    1,  // SBT stride
                    0,  // missSBTIndex
                    u0, u1);

                // TODO: later, find pdf for each material, using default for now
                // heuristics uses next direction & sun direction pdfs

                if (prd.foundLightSource) {
                    float pdf_material = INV_PI * dot(random_d, prd.isect.newDir);
                    float3 col = powerHeuristics(1, 1.f, 1, pdf_material) * prd.rayColor;
                    prd.foundLightSource = false;
                    prd.pixelColor += col * prd.rayColor;
                }
                
            }
            
        }

        if (!prd.isDone) // reached max depth and didn't hit a light
                         // TODO: sample direct lighting at this point
        {
            // Direct Lighting
            prd.isect.newDir = sampleSun(rng2(prd.seed));
            optixTrace(params.rootHandle,
                prd.isect.pos,
                prd.isect.newDir,
                0.f,    // tmin
                1e20f,  // tmax
                0.0f,   // rayTime
                OptixVisibilityMask(255),
                OPTIX_RAY_FLAG_NONE,  // OPTIX_RAY_FLAG_NONE,
                0,  // SBT offset
                1,  // SBT stride
                0,  // missSBTIndex
                u0, u1);
            if (!prd.foundLightSource) {
                prd.pixelColor = make_float3(0.f);
            }
        }

        finalColor += prd.pixelColor;
        finalAlbedo += prd.pixelAlbedo;
        finalNormal += prd.pixelNormal;
    }

    finalColor /= NUM_SAMPLES;
    finalAlbedo /= NUM_SAMPLES;
    finalNormal /= NUM_SAMPLES;

    // accumulate colors
    const uint32_t fbIndex = ix + iy * params.windowSize.x;

    float4 cumColor = make_float4(finalColor, 1.f);

    int frameId = params.frame.frameId;
    if (frameId > 0) {
        cumColor += float(frameId) * params.frame.colorBuffer[fbIndex];
        cumColor /= (frameId + 1.f);
    }

    params.frame.colorBuffer[fbIndex] = cumColor;
}

static __forceinline__ __device__
const ChunkData& getChunkData()
{
    return *(const ChunkData*)optixGetSbtDataPointer();
}

static __forceinline__ __device__
void getVerts(const ChunkData& chunkData, Vertex* v1, Vertex* v2, Vertex* v3)
{
    const int primID = optixGetPrimitiveIndex();
    const uint3 vIdx = chunkData.idx[primID];
    *v1 = chunkData.verts[vIdx.x];
    *v2 = chunkData.verts[vIdx.y];
    *v3 = chunkData.verts[vIdx.z];
}

static __forceinline__ __device__
float3 getBarycentricCoords()
{
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;
    return make_float3(1.f - u - v, u, v);
}

extern "C" __global__ void __miss__radiance()
{
    const float3 rayDir = optixGetWorldRayDirection();
    PRD& prd = *getPRD<PRD>();

    float3 skyColor;
    float d = dot(rayDir, params.sunDir);
    if (d > 0.99f)
    {
        float hue = dot(params.sunDir, make_float3(0.f, 1.f, 0.f));
        skyColor = make_float3(1.0f, 0.6f + 0.2f * hue, 0.4f + 0.2f * hue) * (1.f - 5000.f * (1.f - d) * (1.f - d));
        prd.foundLightSource = true;
    }
    else
    {
        skyColor = make_float3(0.5f, 0.8f, 1.0f) * 0.2f;
    }

    prd.pixelColor += skyColor * prd.rayColor;
    prd.isDone = true;
}

extern "C" __global__ void __closesthit__radiance() {
    PRD& prd = *getPRD<PRD>();

    const ChunkData& chunkData = getChunkData();
    Vertex v1, v2, v3;
    getVerts(chunkData, &v1, &v2, &v3);

    const float3 bary = getBarycentricCoords();
    float2 uv = bary.x * v1.uv + bary.y * v2.uv + bary.z * v3.uv;
    float4 diffuseCol = tex2D<float4>(chunkData.tex_diffuse, uv.x, uv.y);

    const float3 rayDir = optixGetWorldRayDirection();
    float3 isectPos = optixGetWorldRayOrigin() + rayDir * optixGetRayTmax();

    float3 nor = normalize(bary.x * v1.nor + bary.y * v2.nor + bary.z * v3.nor);
    float3 newDir = calculateRandomDirectionInHemisphere(nor, rng2(prd.seed));

    // don't multiply by lambert term since it's canceled out by PDF for uniform hemisphere sampling

    // prd.pixelColor *= prd.rayColor;
    prd.rayColor *= make_float3(diffuseCol);
    prd.isect.pos = isectPos + nor * 0.001f;
    prd.isect.newDir = newDir;
}

extern "C" __global__ void __anyhit__radiance()
{
    const ChunkData& chunkData = getChunkData();
    Vertex v1, v2, v3;
    getVerts(chunkData, &v1, &v2, &v3);

    const float3 bary = getBarycentricCoords();
    float2 uv = bary.x * v1.uv + bary.y * v2.uv + bary.z * v3.uv;
    float4 diffuseCol = tex2D<float4>(chunkData.tex_diffuse, uv.x, uv.y);

    if (diffuseCol.w == 0.f)
    {
        optixIgnoreIntersection();
    }
    // TODO: figure out whether to use normal faceforwards before re-enabling this
    //else
    //{
    //    PRD& prd = *getPRD<PRD>();
    //    if (rng(prd.seed) >= diffuseCol.w)
    //    {
    //        optixIgnoreIntersection();
    //    }
    //}
}

extern "C" __global__ void __anyhit__shadow() {
    PRD& prd = *getPRD<PRD>();
    prd.foundLightSource = false;
    optixTerminateRay();
}

extern "C" __global__ void __exception__all()
{
    // This assumes that the launch dimensions are matching the size of the output buffer.

    const uint3 theLaunchIndex = optixGetLaunchIndex();

    const int theExceptionCode = optixGetExceptionCode();
    printf("Exception %d at (%u, %u)\n", theExceptionCode, theLaunchIndex.x, theLaunchIndex.y);
}