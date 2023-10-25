#version 330

#include defines.glsl

const float sunlightStrength = 1.f;
const vec3 ambientLight = vec3(0.8, 0.98, 1.0) * 0.12f;

uniform sampler2D tex_blockDiffuse;
uniform sampler2DShadow tex_shadowMap;
uniform sampler3D tex_volume;

uniform vec3 u_sunDir;

in vec3 fs_pos;
in vec3 fs_nor;
in vec2 fs_uv;
in vec4 fs_lightSpacePos;
in vec4 fs_volumePos;

out vec4 fragColor;

#define NUM_SHADOW_SAMPLES 16
vec2 poissonDisk[NUM_SHADOW_SAMPLES] = vec2[](
    vec2(-0.94201624, -0.39906216),
    vec2(0.94558609, -0.76890725),
    vec2(-0.094184101, -0.92938870),
    vec2(0.34495938, 0.29387760),
    vec2(-0.91588581, 0.45771432),
    vec2(-0.81544232, -0.87912464),
    vec2(-0.38277543, 0.27676845),
    vec2(0.97484398, 0.75648379),
    vec2(0.44323325, -0.97511554),
    vec2(0.53742981, -0.47373420),
    vec2(-0.26496911, -0.41893023),
    vec2(0.79197514, 0.19090188),
    vec2(-0.24188840, 0.99706507),
    vec2(-0.81409955, 0.91437590),
    vec2(0.19984126, 0.78641367),
    vec2(0.14383161, -0.14100790)
);

#define POISSON_DISK_SIZE 0.0001f

float calculateShadow() {
    vec3 shadowCoords = fs_lightSpacePos.xyz / fs_lightSpacePos.w;
    shadowCoords = (shadowCoords + 1.f) * 0.5f;

    float visibility = 1.0;
    const float visiblityPerSample = 1.f / NUM_SHADOW_SAMPLES;
    for (int i = 0; i < NUM_SHADOW_SAMPLES; ++i) {
        vec3 diskCoords = vec3(shadowCoords.xy + poissonDisk[i] * POISSON_DISK_SIZE, shadowCoords.z);
        visibility -= visiblityPerSample * (1.f - texture(tex_shadowMap, diskCoords));
    }

    return visibility;
}  

void main() {
    vec4 diffuseCol = texture(tex_blockDiffuse, fs_uv);

    float nightFactor = smoothstep(-0.1f, 0.1f, u_sunDir.y);

    float lambert = max(dot(fs_nor, u_sunDir), 0.0);
    lambert *= nightFactor;

    float sunVisibility = calculateShadow();
    vec3 finalColor = (ambientLight + (lambert * sunlightStrength * sunVisibility)) * diffuseCol.rgb;

    vec4 scatteringInformation = texture(tex_volume, vec3(fs_volumePos));
    vec3 inScattering = scatteringInformation.rgb;
    float transmittance = scatteringInformation.a;

    vec3 colorWithFog = finalColor.rgb * transmittance + inScattering;

    float fogFactor = 0.5f * clamp(1 - dot(normalize(vec3(u_sunDir)), vec3(0, 1, 0)), 0, 1);
    finalColor = mix(finalColor, colorWithFog, fogFactor * nightFactor);

    fragColor = vec4(finalColor, 1.f);
}
