#version 330

uniform mat4 u_viewProjMat;

in vec3 v_pos;

void main() {
    gl_Position = u_viewProjMat * vec4(v_pos, 1.f);
}
