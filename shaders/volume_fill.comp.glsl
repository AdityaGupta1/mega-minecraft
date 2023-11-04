#version 430

layout(local_size_x = 320, local_size_y = 1, local_size_z = 1) in;
layout(rgba16f) uniform image3D img_volume;

uniform sampler2DShadow tex_shadowMap;

//uniform vec4 u_sunDir;
//uniform vec4 u_moonDir;
uniform mat4 u_sunViewProjMat;

uniform mat4 u_viewProjMat;
uniform mat4 u_invViewProjMat;

uniform mat4 u_invViewMat;
uniform mat4 u_projMat;

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
    return mix(0.0f, 0.6f, clamp(exponentialDecay, 0.2f, 1.0f));
}

//float getPhaseFunction(float cosPhi, float gFactor)
//{
//    float gFactor2 = gFactor * gFactor;
//    return (1 - gFactor2) / pow(abs(1 + gFactor2 - 2 * gFactor * cosPhi), 1.5f) * PI_OVER_FOUR;
//}

const vec3 sunColor = vec3(0.9922, 0.9843, 0.8275);
const vec3 sunOppositeColor = vec3(0.8275, 0.8354, 0.9922);

vec3 getSunLighting(vec3 worldPos, vec3 viewDirection)
{
    vec4 lightSpacePos = u_sunViewProjMat * vec4(worldPos, 1);
    vec3 shadowCoords = lightSpacePos.xyz / lightSpacePos.w;
    shadowCoords = (shadowCoords + 1.f) * 0.5f;
    float visibility = texture(tex_shadowMap, shadowCoords);

    //float cosPhi = dot(u_sunDir.xyz, viewDirection);
    //float sunPhaseFunctionValue = getPhaseFunction(cosPhi, 0); // last value = anisotropy
    float sunPhaseFunctionValue = PI_OVER_FOUR;

    //vec3 sunMixColor = mix(sunOppositeColor, sunColor, (cosPhi + 1.f) * 0.5);
    //return visibility * sunMixColor * sunPhaseFunctionValue;
    return visibility * sunColor * sunPhaseFunctionValue;
}

void main()
{
    vec3 screenCoords = screenCoordsFromThreadPos(gl_GlobalInvocationID.xyz);
    float linearDepth = volumeZPosToDepth(screenCoords.z);

    const vec3 camPos = u_invViewMat[3].xyz;
    const vec3 camForward = -vec3(u_invViewMat[2]);
    const vec3 camRight = vec3(u_invViewMat[0]);
    const vec3 camUp = vec3(u_invViewMat[1]);
    vec3 worldDir = normalize(camForward + (screenCoords.x / u_projMat[0][0] * camRight) + (screenCoords.y / u_projMat[1][1] * camUp));
    vec3 worldPos = camPos + worldDir * linearDepth;

    float layerThickness = volumeZPosToDepth(screenCoords.z + (1.f / 128.f)) - linearDepth;

    float dustDensity = getDensity(vec3(worldPos));
    float scattering = 0.03 * dustDensity * layerThickness;
    //float absorption = 0.0f;

    vec3 viewDirection = normalize(vec3(worldPos) - vec3(camPos));

    vec3 lighting = getSunLighting(vec3(worldPos), viewDirection);
    lighting *= u_fogColor;

    //vec4 finalOutValue = vec4(lighting * scattering, scattering + absorption);
    vec4 finalOutValue = vec4(lighting * scattering, scattering);
    imageStore(img_volume, ivec3(gl_GlobalInvocationID.xyz), finalOutValue);
}