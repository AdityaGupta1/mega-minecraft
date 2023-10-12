#version 330

in vec3 v_Pos;

void main() {
    gl_Position = vec4(v_Pos, 1.f);
}
