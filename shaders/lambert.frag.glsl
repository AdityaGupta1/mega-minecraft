#version 330

const float sunlightStrength = 1.f;
const vec3 ambientLight = vec3(0.8, 0.98, 1.0) * 0.16f;

uniform sampler2D tex_blockDiffuse;
uniform sampler2DShadow tex_shadowMap;

uniform vec3 u_sunDir;

in vec3 fs_nor;
in vec2 fs_uv;
in vec4 fs_lightPosSpace;

out vec4 fragColor;

float calculateShadow() {
    vec3 projCoords = fs_lightPosSpace.xyz / fs_lightPosSpace.w;
    projCoords = (projCoords + 1.f) / 2.f;
    return texture(tex_shadowMap, projCoords);
}  

void main() {
    vec4 diffuseCol = texture(tex_blockDiffuse, fs_uv);

    float lambert = max(dot(fs_nor, u_sunDir), 0.0);
    lambert *= smoothstep(-0.1f, 0.1f, u_sunDir.y);

    float sunVisibility = calculateShadow();
    vec3 finalColor = (ambientLight + (lambert * sunlightStrength * sunVisibility)) * diffuseCol.rgb;

    fragColor = vec4(finalColor, 1.f);
}
