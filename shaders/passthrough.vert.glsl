#version 330

in vec3 vs_pos;

void main() {
    gl_Position = vec4(vs_pos, 1.f);
}
