#version 330

uniform sampler2D tex_blockDiffuse;

in vec2 fs_uv;

out vec4 fragColor;

void main() {
    vec4 diffuseCol = texture(tex_blockDiffuse, fs_uv);

    fragColor = vec4(diffuseCol.rgb, 1);
}
