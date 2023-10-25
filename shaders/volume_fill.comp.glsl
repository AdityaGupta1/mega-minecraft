#version 430

#define PI           3.1415926535897932384626f
#define PI_OVER_FOUR 0.7853981633974483096156f

layout(local_size_x = 320, local_size_y = 1, local_size_z = 1) in;
layout(rgba16f) uniform image3D img_volume;

uniform sampler2DShadow tex_shadowMap;

uniform vec4 u_sunDir;
uniform mat4 u_sunViewProjMat;

uniform mat4 u_viewProjMat;
uniform mat4 u_invViewProjMat;

uniform mat4 u_viewMat;
uniform mat4 u_invViewMat;

uniform vec3 u_fogColor;

vec3 screenCoordsFromThreadPos(vec3 threadPos)
{
    return threadPos.xyz * vec3(2f / 320, 2f / 180, 1f / 128) + vec3(-1, -1, 0);
}

float volumeZPosToDepth(float volumePosZ)
{
    return (volumePosZ * volumePosZ) * 160;
}

float getDensity(vec3 worldPos)
{
    float exponentialDecay = exp(-0.05f * (worldPos.y - 63));
    return mix(0.f, 0.6f, min(exponentialDecay, 1.0f));
}

float getPhaseFunction(float cosPhi, float gFactor)
{
    float gFactor2 = gFactor * gFactor;
    return (1 - gFactor2) / pow(abs(1 + gFactor2 - 2 * gFactor * cosPhi), 1.5f) * PI_OVER_FOUR;
}

//#define NUM_SHADOW_SAMPLES 16
//vec2 poissonDisk[NUM_SHADOW_SAMPLES] = vec2[](
//    vec2(-0.94201624, -0.39906216),
//    vec2(0.94558609, -0.76890725),
//    vec2(-0.094184101, -0.92938870),
//    vec2(0.34495938, 0.29387760),
//    vec2(-0.91588581, 0.45771432),
//    vec2(-0.81544232, -0.87912464),
//    vec2(-0.38277543, 0.27676845),
//    vec2(0.97484398, 0.75648379),
//    vec2(0.44323325, -0.97511554),
//    vec2(0.53742981, -0.47373420),
//    vec2(-0.26496911, -0.41893023),
//    vec2(0.79197514, 0.19090188),
//    vec2(-0.24188840, 0.99706507),
//    vec2(-0.81409955, 0.91437590),
//    vec2(0.19984126, 0.78641367),
//    vec2(0.14383161, -0.14100790)
//    );
//
//#define POISSON_DISK_SIZE 0.0001f
//
//float calculateShadow(vec3 worldPos)
//{
//    vec4 lightSpacePos = u_sunViewProjMat * vec4(worldPos, 1);
//    vec3 shadowCoords = lightSpacePos.xyz / lightSpacePos.w;
//    shadowCoords = (shadowCoords + 1.f) * 0.5f;
//
//    float visibility = 1.0;
//    const float visiblityPerSample = 1.f / NUM_SHADOW_SAMPLES;
//    for (int i = 0; i < NUM_SHADOW_SAMPLES; ++i)
//    {
//        vec3 diskCoords = vec3(shadowCoords.xy + poissonDisk[i] * POISSON_DISK_SIZE, shadowCoords.z);
//        visibility -= visiblityPerSample * (1.f - texture(tex_shadowMap, diskCoords));
//    }
//
//    return visibility;
//}

vec3 getSunLighting(vec3 worldPos, vec3 viewDirection)
{
    vec4 lightSpacePos = u_sunViewProjMat * vec4(worldPos, 1);
    vec3 shadowCoords = lightSpacePos.xyz / lightSpacePos.w;
    shadowCoords = (shadowCoords + 1.f) * 0.5f;
    float visibility = texture(tex_shadowMap, shadowCoords);

    //float sunPhaseFunctionValue = getPhaseFunction(dot(vec3(u_sunDir), viewDirection), 0); // last value = anisotropy
    float sunPhaseFunctionValue = PI_OVER_FOUR;

    return visibility * vec3(0.9922, 0.9843, 0.8275) * sunPhaseFunctionValue; // static sun color for now
}

void main()
{
    vec3 screenCoords = screenCoordsFromThreadPos(gl_GlobalInvocationID.xyz);
    float linearDepth = volumeZPosToDepth(screenCoords.z);

    // use matrix, change NDC, invert matrix
    // TODO: make this less stupid
    // =======================================
    const vec3 cameraPos = u_invViewMat[3].xyz;
    const vec3 cameraForward = -vec3(u_viewMat[0][2], u_viewMat[1][2], u_viewMat[2][2]);

    vec4 depthWorldPos = vec4(cameraPos + (cameraForward * linearDepth), 1);
    vec4 depthNDCPos = u_viewProjMat * depthWorldPos;
    vec4 actualNDCPos = vec4(screenCoords.xy * depthNDCPos.w, depthNDCPos.zw);
    vec4 worldPos = u_invViewProjMat * actualNDCPos;
    worldPos /= worldPos.w;
    // =======================================

    float layerThickness = volumeZPosToDepth(screenCoords.z + (1.f / 128.f)) - linearDepth;

    float dustDensity = getDensity(vec3(worldPos));
    float scattering = 0.03 * dustDensity * layerThickness;
    //float absorption = 0.0f;

    vec3 viewDirection = normalize(vec3(worldPos) - vec3(cameraPos));

    vec3 lighting = getSunLighting(vec3(worldPos), viewDirection);
    lighting *= u_fogColor;

    //vec4 finalOutValue = vec4(lighting * scattering, scattering + absorption);
    vec4 finalOutValue = vec4(lighting * scattering, scattering);
    imageStore(img_volume, ivec3(gl_GlobalInvocationID.xyz), finalOutValue);
}