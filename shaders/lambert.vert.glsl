#version 330

uniform mat4 u_viewProjMat;

in vec3 vs_pos;

out vec3 fs_col;

vec3 rand(vec3 v) {
    return fract(sin(vec3(
        dot(v, vec3(265.52, 401.19, 387.90)),
        dot(v, vec3(759.03, 772.77, 344.12)),
        dot(v, vec3(564.13, 466.08, 762.51))
    )) * 43758.5453);
}

void main() {
    gl_Position = u_viewProjMat * vec4(vs_pos, 1);

    fs_col = rand(vs_pos);
}
