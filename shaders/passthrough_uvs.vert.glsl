#version 330

in vec3 vs_pos;
in vec2 vs_uv;

out vec2 fs_uv;

void main() {
    gl_Position = vec4(vs_pos, 1.f);
    fs_uv = vs_uv;
}
