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

#define DO_RUSSIAN_ROULETTE 1

#define NUM_SAMPLES 2
#define MAX_RAY_DEPTH 4

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

static __forceinline__ __device__ float luminance(float3 color)
{
    return dot(color, make_float3(0.2126, 0.7152, 0.0722));
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
    const float3 normal = normalize(params.sunDir);

    const float3 perpendicularDirection1 = normalize(cross(normal, calculateDirectionNotNormal(normal)));
    const float3 perpendicularDirection2 = normalize(cross(normal, perpendicularDirection1));

    const float around = sample.y * TWO_PI; // theta

    float3 dir = normalize(cos(around) * perpendicularDirection1 + sin(around) * perpendicularDirection2);

    // 0.100 is the max radius to sample within the sun at dot = 0.995
    return normalize(normal + sample.x * 0.100f * dir);
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

    #pragma unroll
    for (int sample = 0; sample < NUM_SAMPLES; ++sample)
    {
        prd.isDone = false;
        prd.needsFirstHitData = true;
        prd.foundLightSource = false;
        prd.rayColor = make_float3(1.f);
        prd.pixelColor = make_float3(0.f);
        prd.isect.pos = camera.position;
        prd.isect.newDir = rayDir;
        prd.specularHit = false;

        #pragma unroll
        for (int depth = 0; depth < MAX_RAY_DEPTH; ++depth)
        {
            // 1. BSDF
            prd.foundLightSource = false;

            optixTrace(params.rootHandle,
                prd.isect.pos,
                prd.isect.newDir,
                0.f,    // tmin
                1e20f,  // tmax
                0.0f,   // rayTime
                OptixVisibilityMask(255),
                OPTIX_RAY_FLAG_NONE,
                0,  // SBT offset
                1,  // SBT stride
                0,  // missSBTIndex
                u0, u1);

            if (prd.isDone)
            {
                break;
            }

            if (prd.specularHit && depth % 2 == 0 && depth < MAX_RAY_DEPTH) {
                --depth;
            }

            // MIS: sample light source
            // 2. pdf from Sun & random point on sun
            if (!prd.specularHit) {
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
                    OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                    1,  // SBT offset
                    1,  // SBT stride
                    0,  // missSBTIndex
                    u0, u1);

                // if specular material is hit and entered - see if we exit it 

                if (prd.specularHit) {
                    optixTrace(params.rootHandle,
                        prd.isect.pos,
                        prd.isect.newDir,
                        0.f,    // tmin
                        1e20f,  // tmax
                        0.0f,   // rayTime
                        OptixVisibilityMask(255),
                        OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                        1,  // SBT offset
                        1,  // SBT stride
                        0,  // missSBTIndex
                        u0, u1);
                }

                // TODO: later, find pdf for each material, using default for now
                // heuristics uses next direction & sun direction pdfs

                if (prd.foundLightSource) {
                    float pdf_material = INV_PI * dot(random_d, prd.isect.newDir);
                    float3 col = powerHeuristics(1, 1.f, 1, pdf_material) * prd.rayColor;
                    prd.foundLightSource = false;
                    prd.pixelColor += col * prd.rayColor;
                }
            }
            

#if DO_RUSSIAN_ROULETTE
            if (depth > 2)
            {
                float q = fmax(0.05f, 1.f - luminance(prd.pixelColor));
                if (rng(prd.seed) < q)
                {
                    prd.pixelColor = make_float3(0);
                    break;
                }

                prd.pixelColor /= (1.f - q);
            }
#endif
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

    int frameId = params.frame.frameId;
    if (frameId > 0) {
        float multiplier = 1.f / (frameId + 1.f);
        finalColor = (finalColor + frameId * make_float3(params.frame.colorBuffer[fbIndex])) * multiplier;
        finalAlbedo = (finalAlbedo + frameId * make_float3(params.frame.albedoBuffer[fbIndex])) * multiplier;
        finalNormal = (finalNormal + frameId * make_float3(params.frame.normalBuffer[fbIndex])) * multiplier;
    }

    params.frame.colorBuffer[fbIndex] = make_float4(finalColor, 1.f);
    params.frame.albedoBuffer[fbIndex] = make_float4(finalAlbedo, 1.f);
    params.frame.normalBuffer[fbIndex] = make_float4(finalNormal, 1.f);
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

static __forceinline__ __device__
float3 refract(float3 wo, float3 n, float eta) {
    float k = 1.f - eta * eta * (1.f - dot(n, wo) * dot(n, wo));
    if (k < 0.f) {
        return make_float3(0.f);
    }
    else {
        return normalize(eta * wo - (eta * dot(n, wo) + sqrt(k)) * n);
    }
}

static __forceinline__ __device__ 
float smoothstep(float edge0, float edge1, float x)
{
    x = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return x * x * (3 - 2 * x);
}

__device__ float3 getSkyColor(float3 rayDir, bool& foundLightSource)
{
    float d = dot(rayDir, params.sunDir);
    float3 skyColor = make_float3(0.f);
    if (d > 0.995f)
    {
        float hue = params.sunDir.y;
        float3 sunColor = make_float3(1.0f, 0.5f + 0.15f * hue, 0.3f + 0.15f * hue) * (1.f - 5000.f * (1.f - d) * (1.f - d));
        skyColor += sunColor * 2.8f;
        foundLightSource = true;
    }
    else
    {
        skyColor += make_float3(0.5f, 0.8f, 1.0f) * 0.2f;
    }

    if (d > 0.98f)
    {
        skyColor += powf(smoothstep(0.98f, 0.9975f, d), 3.f) * make_float3(1.0f, 0.7f, 0.5f) * 0.4f;
    }

    return skyColor;
}

extern "C" __global__ void __miss__radiance()
{
    const float3 rayDir = optixGetWorldRayDirection();
    PRD& prd = *getPRD<PRD>();

    float3 skyColor = getSkyColor(rayDir, prd.foundLightSource);

    prd.pixelColor += skyColor * prd.rayColor;
    if (prd.specularHit && prd.foundLightSource) {
        prd.pixelColor += skyColor * prd.rayColor * 2;
    }
    prd.isDone = true;

    if (prd.needsFirstHitData)
    {
        prd.needsFirstHitData = false;
        prd.pixelAlbedo = skyColor;
        prd.pixelNormal = -rayDir;
    }
}

static __device__ float schlickFresnel(const float3& V, const float3& N, const float ior)
{
    float cosTheta = fabsf(dot(V, N));
    float R0 = (1 - ior) / (1 + ior);
    R0 = R0 * R0;
    return R0 + (1 - R0) * pow(1.f - cosTheta, 5.f);
}

// noise code from here: https://forum.pjrc.com/index.php?threads/im-looking-for-a-performant-perlin-or-open-simplex-noise-implementation.72409/post-322480
__constant__ unsigned char permutation[] = {
     151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140, 36, 103,
     30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148, 247, 120, 234, 75, 0, 26, 197,
     62, 94, 252, 219, 203, 117, 35, 11, 32, 57, 177, 33, 88, 237, 149, 56, 87, 174, 20,
     125, 136, 171, 168, 68, 175, 74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231,
     83, 111, 229, 122, 60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102,
     143, 54, 65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169, 200,
     196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64, 52, 217, 226,
     250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212, 207, 206, 59, 227, 47, 16,
     58, 17, 182, 189, 28, 42, 223, 183, 170, 213, 119, 248, 152, 2, 44, 154, 163, 70, 221,
     153, 101, 155, 167, 43, 172, 9, 129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232,
     178, 185, 112, 104, 218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179,
     162, 241, 81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157,
     184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93, 222, 114,
     67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180
};

__device__ float fade(float t) { return t * t * t * (t * (t * 6 - 15) + 10); }
__device__ float lerp2(float t, float a, float b) { return a + t * (b - a); }
__device__ float grad(int hash, float x, float y, float z)
{
    // convert lower 4 bits of hash code into 12 gradient directions
    int    h = hash & 15;
    float  u = h < 8 ? x : y,
           v = h < 4 ? y : h == 12 || h == 14 ? x : z;
    return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
}

#define P(x) permutation[(x) & 255]

__device__ float pnoise(float3 p)
{
    // find unit cube containing point
    int X = (int)floorf(p.x) & 255,
        Y = (int)floorf(p.y) & 255,
        Z = (int)floorf(p.z) & 255;

    // find relative x, y, z, of each point in cube
    p.x -= floorf(p.x);
    p.y -= floorf(p.y);
    p.z -= floorf(p.z);

    // compute fade curves for each of x, y, z
    float u = fade(p.x),
          v = fade(p.y),
          w = fade(p.z);

    // hash coordinates of 8 cube corners
    int A  = P(X) + Y,
        AA = P(A) + Z,
        AB = P(A + 1) + Z,
        B  = P(X + 1) + Y,
        BA = P(B) + Z,
        BB = P(B + 1) + Z;

    // add blended results from 8 corners of cube
    return
        lerp2(w,
            lerp2(v,
                lerp2(u,
                    grad(P(AA), p.x, p.y, p.z),
                    grad(P(BA), p.x - 1, p.y, p.z)),
                lerp2(u,
                    grad(P(AB), p.x, p.y - 1, p.z),
                    grad(P(BB), p.x - 1, p.y - 1, p.z))),
            lerp2(v,
                lerp2(u,
                    grad(P(AA + 1), p.x, p.y, p.z - 1),
                    grad(P(BA + 1), p.x - 1, p.y, p.z - 1)),
                lerp2(u,
                    grad(P(AB + 1), p.x, p.y - 1, p.z - 1),
                    grad(P(BB + 1), p.x - 1, p.y - 1, p.z - 1)
                )
            )
        );
}

#undef P

float __device__ fbm(float3 p)
{
    float fbm = 0.f;
    float amplitude = 1.f;
    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        amplitude *= 0.5f;
        fbm += amplitude * pnoise(p);
        p *= 2.f;
    }
    return fbm;
}

__device__ void applyWaveNoise(const float3& pos, float3& nor)
{
    float3 noisePos = pos * 0.45f;
    noisePos.x *= 2.2f;

    float perlinX = fbm(noisePos);
    float perlinZ = fbm(noisePos + make_float3(74159.21f, 21982.43f, 18923.34f));

    nor.x += perlinX * 0.3f;
    nor.z += perlinZ * 0.3f;
    nor = normalize(nor);
}

extern "C" __global__ void __closesthit__radiance() {
    PRD& prd = *getPRD<PRD>();

    const ChunkData& chunkData = getChunkData();
    Vertex v1, v2, v3;
    getVerts(chunkData, &v1, &v2, &v3);

    const float3 bary = getBarycentricCoords();
    float2 uv = bary.x * v1.uv + bary.y * v2.uv + bary.z * v3.uv;
    float3 nor = normalize(bary.x * v1.nor + bary.y * v2.nor + bary.z * v3.nor);

    const float3 rayDir = optixGetWorldRayDirection();
    float3 diffuseCol = make_float3(tex2D<float4>(chunkData.tex_diffuse, uv.x, uv.y));

    const float3 rayOrigin = optixGetWorldRayOrigin();
    const float3 isectPos = rayOrigin + rayDir * optixGetRayTmax();

    // REFL REFR

    // ENTERING: mix of REFR at dot = 1 to REFL at dot = 0
    // EXITING: all REFL at dot = 0 to sinThetaT = 1, and mix of REFL to REFR at dot = 1

    if (v1.m.reflecting && v1.m.refracting) {
        prd.specularHit = true;
        prd.needsFirstHitData = false;
        float ior = v1.m.ior;

        if (v1.m.wavy) {
            applyWaveNoise(isectPos, nor);
        }
        
        float entering = dot(rayDir, nor);

        if (entering < 0.f) {
            if (rng(prd.seed) < -entering) {
                // ENTERING REFR
                prd.isect.pos = isectPos - nor * 0.001f;
                prd.isect.newDir = refract(rayDir, nor, 1.f / ior);

                float fresnel = schlickFresnel(rayDir, nor, ior);
                prd.rayColor *= (1.f - fresnel);
            }
            else {
                // ENTERING REFL
                prd.isect.pos = isectPos + nor * 0.001f;
                prd.isect.newDir = reflect(rayDir, nor);

                float fresnel = schlickFresnel(rayDir, nor, ior);
                prd.rayColor *= fresnel;
            }
        }
        else {
            float sinThetaT = ior * sqrt(1 - entering * entering);

            if (rng(prd.seed) < entering / fmax(1.f, sinThetaT)) {
                // EXITING REFR
                prd.isect.pos = isectPos + nor * 0.001f;
                prd.isect.newDir = refract(rayDir, -nor, ior);

                float fresnel = schlickFresnel(rayDir, nor, ior);
                prd.rayColor *= (1.f - fresnel);
            }
            else {
                // EXITING REFL
                prd.isect.pos = isectPos - nor * 0.001f;
                prd.isect.newDir = reflect(rayDir, -nor);

                float fresnel = schlickFresnel(rayDir, -nor, ior);
                prd.rayColor *= fresnel;
            }
            
        }
        prd.rayColor *= ior * diffuseCol;
        return;
    }

    // EMISSION
    prd.specularHit = false;
    if (dot(rayDir, nor) > 0.f)
    {
        nor = -nor;
    }
    float3 newDir = calculateRandomDirectionInHemisphere(nor, rng2(prd.seed));

    if (diffuseCol.x == 0.f && diffuseCol.y == 0.f && diffuseCol.z == 0.f)
    {
        float4 emissiveTexCol = tex2D<float4>(chunkData.tex_emissive, uv.x, uv.y);
        if (emissiveTexCol.w > 0.f)
        {
            float indirectStrength = emissiveTexCol.w * 100.f;
            float3 emissiveCol = make_float3(emissiveTexCol) * (prd.needsFirstHitData ? 1.5f : indirectStrength); // make indirect emissive lighting much stronger

            prd.pixelColor += prd.rayColor * emissiveCol;

            if (prd.needsFirstHitData)
            {
                prd.needsFirstHitData = false;
                prd.pixelAlbedo = emissiveCol;
                prd.pixelNormal = nor;
            }

            prd.isDone = true;
            return;
        }
    }


    // don't multiply by lambert term since it's canceled out by PDF for uniform hemisphere sampling

    prd.rayColor *= diffuseCol;
    prd.isect.pos = isectPos + nor * 0.001f;
    prd.isect.newDir = newDir;

    if (prd.needsFirstHitData)
    {
        prd.needsFirstHitData = false;
        prd.pixelAlbedo = diffuseCol;
        prd.pixelNormal = nor;
    }
}

// returns true if ray should continue
// also DOES NOT multiply ray color for semi-transparent objects, since I'm pretty sure anyhit can activate for intersections past the closest one
static __device__ bool anyhitAlphaTest()
{
    const ChunkData& chunkData = getChunkData();
    Vertex v1, v2, v3;
    getVerts(chunkData, &v1, &v2, &v3);

    const float3 bary = getBarycentricCoords();
    float2 uv = bary.x * v1.uv + bary.y * v2.uv + bary.z * v3.uv;
    float4 diffuseCol = tex2D<float4>(chunkData.tex_diffuse, uv.x, uv.y);

    const float alpha = diffuseCol.w;

    if (alpha == 0.f)
    {
        return true;
    }
    else
    {
        return false;
    }
}

extern "C" __global__ void __anyhit__radiance()
{
    if (anyhitAlphaTest())
    {
        optixIgnoreIntersection();
    }
}

extern "C" __global__ void __anyhit__shadow()
{
    PRD& prd = *getPRD<PRD>();

    const ChunkData& chunkData = getChunkData();
    Vertex v1, v2, v3;
    getVerts(chunkData, &v1, &v2, &v3);

    const float3 bary = getBarycentricCoords();
    float2 uv = bary.x * v1.uv + bary.y * v2.uv + bary.z * v3.uv;
    float3 nor = normalize(bary.x * v1.nor + bary.y * v2.nor + bary.z * v3.nor);

    const float3 rayDir = optixGetWorldRayDirection();
    float3 diffuseCol = make_float3(tex2D<float4>(chunkData.tex_diffuse, uv.x, uv.y));

    const float3 rayOrigin = optixGetWorldRayOrigin();
    const float3 isectPos = rayOrigin + rayDir * optixGetRayTmax();

    // REFL REFR

    // ENTERING: mix of REFR at dot = 1 to REFL at dot = 0
    // EXITING: all REFL at dot = 0 to sinThetaT = 1, and mix of REFL to REFR at dot = 1

    //if (v1.m.reflecting && v1.m.refracting) {
    //    prd.specularHit = true;
    //    prd.needsFirstHitData = false;
    //    float ior = v1.m.ior;

    //    if (v1.m.wavy)
    //    {
    //        applyWaveNoise(isectPos, nor);
    //    }

    //    float entering = dot(rayDir, nor);

    //    if (entering < 0.f) {
    //        if (rng(prd.seed) < -entering) {
    //            // ENTERING REFR
    //            prd.isect.pos = isectPos - nor * 0.001f;
    //            prd.isect.newDir = refract(rayDir, nor, 1.f / ior);

    //            float fresnel = schlickFresnel(rayDir, nor, ior);
    //            prd.rayColor *= (1.f - fresnel);
    //            return;
    //        }
    //        else {
    //            prd.foundLightSource = false;
    //            optixTerminateRay();
    //        }
    //    }
    //    else {
    //        float sinThetaT = v1.m.ior * sqrt(1 - entering * entering) + 0.0001f;

    //        if (rng(prd.seed) < entering / fmax(1.f, sinThetaT)) {
    //            // EXITING REFR
    //            if (dot(params.sunDir, refract(rayDir, -nor, ior)) > 0.99f) {
    //                float fresnel = schlickFresnel(rayDir, nor, ior);
    //                prd.rayColor *= (1.f - fresnel);
    //                prd.rayColor *= 0.5f / ior * diffuseCol;
    //                optixIgnoreIntersection();
    //                return;
    //            }
    //        }
    //        else {
    //            // EXITING REFL
    //            prd.foundLightSource = false;
    //            optixTerminateRay();
    //        }
    //    }
    //}
    //else if (anyhitAlphaTest())
    //{
    //    optixIgnoreIntersection();
    //    return;
    //}
    if (anyhitAlphaTest())
    {
        optixIgnoreIntersection();
        return;
    }

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