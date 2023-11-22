#version 330

uniform sampler2D tex_bufColor;

in vec2 fs_uv;

out vec3 out_color;

vec3 ACESFilm(vec3 x) {
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.f, 1.f);
}

vec3 hdrToLdr(vec3 col) {
    return pow(ACESFilm(col), vec3(INVERSE_GAMMA));
}

void main() {
    out_color = hdrToLdr(texture(tex_bufColor, fs_uv).rgb);
}
