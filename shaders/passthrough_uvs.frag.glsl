#version 330

in vec2 fs_uv;

out vec4 fragColor;

void main() {
    fragColor = vec4(fs_uv, 0.f, 1.f);
}
