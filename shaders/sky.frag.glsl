#version 330

uniform mat4 u_invViewProjMat;

in vec2 fs_uv;

out vec4 fragColor;

void main() {
    vec2 ndc = fs_uv * 2.f - 1.f;
    vec3 worldDir = normalize(vec3(u_invViewProjMat * vec4(ndc, 1.f, 1.f)));

    fragColor = vec4(worldDir, 1.f);
}
