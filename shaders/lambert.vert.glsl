#version 330

uniform mat4 u_modelMat;
uniform mat4 u_viewProjMat;
uniform mat4 u_sunViewProjMat;

in vec3 vs_pos;
in vec3 vs_nor;
in vec2 vs_uv;

out vec3 fs_pos;
out vec3 fs_nor;
out vec2 fs_uv;
out vec4 fs_lightSpacePos;
out vec4 fs_volumePos;

//vec3 rand(vec3 v) {
//    return fract(sin(vec3(
//        dot(v, vec3(265.52, 401.19, 387.90)),
//        dot(v, vec3(759.03, 772.77, 344.12)),
//        dot(v, vec3(564.13, 466.08, 762.51))
//    )) * 43758.5453);
//}

float depthToVolumeZPos(float depth) {
    return sqrt(abs(depth / 160));
}

void main() {
    vec4 modelPos = u_modelMat * vec4(vs_pos, 1);
    gl_Position = u_viewProjMat * modelPos;

    fs_pos = modelPos.xyz;
    fs_nor = vs_nor;
    fs_uv = vs_uv;
    fs_lightSpacePos = u_sunViewProjMat * vec4(modelPos.xyz / modelPos.w, 1);

    fs_volumePos = vec4((gl_Position.xy / gl_Position.w + 1) * 0.5, depthToVolumeZPos(gl_Position.z), 1);
}
