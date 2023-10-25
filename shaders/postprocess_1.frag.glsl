#version 330

uniform sampler2D tex_bufColor;

in vec2 fs_uv;

layout(location = 0) out vec3 out_color;
layout(location = 1) out vec3 out_bloomColor;

float luminance(vec3 color) {
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

void main() {
    out_color = texture(tex_bufColor, fs_uv).rgb;
    out_bloomColor = luminance(out_color) > 1.f ? out_color : vec3(0);
}
