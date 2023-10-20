#version 330

const vec3 lightDir = normalize(vec3(5, 7, 3));
const float lightStrength = 1.f;
const vec3 ambientLight = vec3(0.8, 0.98, 1.0) * 0.2f;

uniform sampler2D tex_blockDiffuse;

in vec3 fs_nor;
in vec2 fs_uv;

out vec4 fragColor;

void main() {
    vec4 diffuseCol = texture(tex_blockDiffuse, fs_uv);

    float NdotL = max(dot(fs_nor, lightDir), 0.0);
    vec3 lambert = diffuseCol.rgb * (ambientLight + (NdotL * lightStrength));

    fragColor = vec4(lambert, 1.f);
}
