#version 330

in vec3 fs_col;

out vec4 fragColor;

void main() {
    fragColor = vec4(fs_col, 1);
}
