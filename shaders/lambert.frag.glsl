#version 330

#define MIN_BIAS 0.005
#define MAX_BIAS 0.05

const float lightStrength = 1.f;
const vec3 ambientLight = vec3(0.8, 0.98, 1.0) * 0.16f;

uniform sampler2D tex_blockDiffuse;
uniform sampler2D tex_shadowMap;

uniform vec3 u_sunDir;

in vec3 fs_nor;
in vec2 fs_uv;
in vec4 fs_lightPosSpace;

out vec4 fragColor;

float calculateShadow() {
    vec3 projCoords = fs_lightPosSpace.xyz / fs_lightPosSpace.w;
    projCoords = (projCoords + 1.f) / 2.f;

    float closestDepth = texture(tex_shadowMap, projCoords.xy).r; 
    float currentDepth = projCoords.z;

    float bias = max(MAX_BIAS * (1.0 - dot(fs_nor, u_sunDir)), MIN_BIAS);  
    float shadow = currentDepth - bias > closestDepth ? 1.0 : 0.0;
    return shadow;
}  

void main() {
    vec4 diffuseCol = texture(tex_blockDiffuse, fs_uv);

    float lambert = max(dot(fs_nor, u_sunDir), 0.0);
    lambert *= smoothstep(-0.1f, 0.1f, u_sunDir.y);

    float shadow = calculateShadow();
    vec3 finalColor = (ambientLight + (lambert * lightStrength * (1.f - shadow))) * diffuseCol.rgb;

    fragColor = vec4(finalColor, 1.f);
}
