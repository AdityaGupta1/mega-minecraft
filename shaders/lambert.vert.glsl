#version 330

uniform mat4 u_modelMat;
uniform mat4 u_viewProjMat;
uniform mat4 u_sunViewProjMat;

in vec3 vs_pos;
in int vs_norUv;

out vec3 fs_nor;
out vec2 fs_uv;
out vec4 fs_lightPosSpace;

vec3 rand(vec3 v) {
    return fract(sin(vec3(
        dot(v, vec3(265.52, 401.19, 387.90)),
        dot(v, vec3(759.03, 772.77, 344.12)),
        dot(v, vec3(564.13, 466.08, 762.51))
    )) * 43758.5453);
}

void main() {
    vec4 modelPos = u_modelMat * vec4(vs_pos, 1);
    gl_Position = u_viewProjMat * modelPos;

    int norZ = vs_norUv & 0x7;
    int norY = (vs_norUv >> 3) & 0x7;
    int norX = (vs_norUv >> 6) & 0x7;
    int uvY = (vs_norUv >> 9) & 0x1FF;
    int uvX = (vs_norUv >> 18) & 0x1FF;

    fs_nor = vec3(0, 1, 0);
    fs_uv = vec2(uvX, uvY) / 256.f;
    fs_lightPosSpace = u_sunViewProjMat * vec4(modelPos.xyz / modelPos.w, 1);
}
