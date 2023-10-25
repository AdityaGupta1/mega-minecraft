#version 330

uniform sampler2D tex_bufBloomColor;

uniform bool u_horizontal;

const float kernel[6] = float[](0.2612, 0.2108, 0.1109, 0.0380, 0.0085, 0.0012);

in vec2 fs_uv;

out vec3 out_bloomColor;

void main()
{
    vec2 texOffset = 1.f / textureSize(tex_bufBloomColor, 0);
    vec3 result = texture(tex_bufBloomColor, fs_uv).rgb * kernel[0];
    if (u_horizontal) {
        for (int i = 0; i < 6; ++i) {
            result += texture(tex_bufBloomColor, fs_uv + vec2(texOffset.x * i, 0.0)).rgb * kernel[i];
            result += texture(tex_bufBloomColor, fs_uv - vec2(texOffset.x * i, 0.0)).rgb * kernel[i];
        }
    } else {
        for (int i = 0; i < 6; ++i) {
            result += texture(tex_bufBloomColor, fs_uv + vec2(0.0, texOffset.y * i)).rgb * kernel[i];
            result += texture(tex_bufBloomColor, fs_uv - vec2(0.0, texOffset.y * i)).rgb * kernel[i];
        }
    }

    out_bloomColor = result.rgb;
}
