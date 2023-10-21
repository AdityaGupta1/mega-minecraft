#version 330

uniform mat4 u_modelMat;
uniform mat4 u_viewProjMat;

in vec3 vs_pos;

void main() {
    gl_Position = u_viewProjMat * u_modelMat * vec4(vs_pos, 1);
}
