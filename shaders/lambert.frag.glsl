#version 330

const float lightStrength = 1.f;
const vec3 ambientLight = vec3(0.8, 0.98, 1.0) * 0.16f;

uniform sampler2D tex_blockDiffuse;

uniform vec3 u_sunDir;

in vec3 fs_nor;
in vec2 fs_uv;

out vec4 fragColor;

void main() {
    vec4 diffuseCol = texture(tex_blockDiffuse, fs_uv);

    float lambert = max(dot(fs_nor, u_sunDir), 0.0);
    lambert *= smoothstep(-0.2f, -0.1f, u_sunDir.y);
    vec3 finalColor = diffuseCol.rgb * (ambientLight + (lambert * lightStrength));

    fragColor = vec4(finalColor, 1.f);
}
