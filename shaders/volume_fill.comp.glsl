#version 430

#define PI 3.1415926535f

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout(rgba16f) uniform image3D img_volume;

uniform sampler2DShadow tex_shadowMap;

uniform vec4 u_sunDir;
uniform mat4 u_sunViewProjMat;

uniform mat4 u_viewProjMat;
uniform mat4 u_invViewProjMat;

uniform mat4 u_viewMat;
uniform mat4 u_invViewMat;

//uniform vec3 u_FogColor;

vec3 screenCoordsFromThreadPos(vec3 threadPos)
{
    return threadPos.xyz * vec3(2f / 320, 2f / 180, 1f / 128) + vec3(-1, -1, 0);
}

float volumeZPosToDepth(float volumePosZ)
{
    return pow(abs(volumePosZ), 2) * 160;
}

float getDensity(vec3 worldPos)
{
    float exponentialDecay = exp(-0.05f * (worldPos.y - 63));
    float delta = min(exponentialDecay, 1.0f);
    return mix(0.3f, 0.6f, delta);
}

float getPhaseFunction(float cosPhi, float gFactor)
{
    float gFactor2 = gFactor * gFactor;
    return (1 - gFactor2) / pow(abs(1 + gFactor2 - 2 * gFactor * cosPhi), 1.5f) * (1.0f / 4.0f * PI);
}

vec3 getSunLighting(vec3 worldPos, vec3 viewDirection)
{
    vec4 lightSpacePos = u_sunViewProjMat * vec4(worldPos, 1);
    vec3 shadowCoords = lightSpacePos.xyz / lightSpacePos.w;
    shadowCoords = (shadowCoords + 1.f) * 0.5f;
    float visibility = texture(tex_shadowMap, shadowCoords);

    float sunPhaseFunctionValue = getPhaseFunction(dot(vec3(u_sunDir), viewDirection), 0); // last value = anisotropy

    return visibility * (vec3(253, 251, 211) / 255.f) * sunPhaseFunctionValue; // static sun color for now
}

void main()
{
    vec3 screenCoords = screenCoordsFromThreadPos(gl_GlobalInvocationID.xyz);
    float linearDepth = volumeZPosToDepth(screenCoords.z);

    // use matrix, change NDC, invert matrix (WHY DOES THIS WORK??????? I'M NOT COMPLAINING BUT STILL????)
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

    float layerThickness = volumeZPosToDepth(screenCoords.z + /*1f / 128*/ 0.0078125f) - linearDepth;

    float dustDensity = getDensity(vec3(worldPos));
    float scattering = 0.03 * dustDensity * layerThickness;
    float absorption = 0.0f;

    vec3 viewDirection = normalize(vec3(worldPos) - vec3(cameraPos));

    vec3 lighting = getSunLighting(vec3(worldPos), viewDirection);
    //lighting *= u_FogColor;
    lighting *= vec3(1.0f, 1.0f, 0.93f); // TODO re-enable fog color uniform

    vec4 finalOutValue = vec4(lighting * scattering, scattering + absorption);
    imageStore(img_volume, ivec3(gl_GlobalInvocationID.xyz), finalOutValue);
}