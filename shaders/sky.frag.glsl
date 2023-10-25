#version 330

uniform sampler3D tex_volume;

uniform mat4 u_invViewMat;
uniform mat4 u_projMat;

uniform vec4 u_sunDir;
uniform vec4 u_moonDir;

in vec2 fs_uv;

out vec4 fragColor;

void main() {
    vec3 camPos = u_invViewMat[3].xyz;
    vec3 camForward = -vec3(u_invViewMat[2]); // not sure why this is backwards but whatever
    vec3 camRight = vec3(u_invViewMat[0]);
    vec3 camUp = vec3(u_invViewMat[1]);
    vec2 ndc = fs_uv * 2.f - 1.f;
    vec3 worldDir = normalize(camForward + (ndc.x / u_projMat[0][0] * camRight) + (ndc.y / u_projMat[1][1] * camUp));
    vec3 worldPos = camPos + 160 * worldDir; // depth = 160 for volumetric fog

    float sunFactor = u_sunDir.w;
    vec3 finalColor;
    if (dot(worldDir, u_sunDir.xyz) > 0.998f) {
        finalColor = vec3(1.0f, 0.8070f, 0.2542f) * 3.f;
    } else if (dot(worldDir, u_moonDir.xyz) > 0.9992f) {
        finalColor = vec3(0.6584f, 0.8879f, 1.0f) * 1.5f;
    } else {
        finalColor = vec3(0.5, 0.8, 1.0) * 0.2 * mix(0.1, 1.0, sunFactor);
    }

    // add volumetric fog
    // -----------------------------------

    vec4 scatteringInformation = texture(tex_volume, vec3(fs_uv, 1));
    vec3 inScattering = scatteringInformation.rgb;
    float transmittance = scatteringInformation.a;

    vec3 colorWithFog = finalColor.rgb * transmittance + inScattering;

    float fogFactor = 0.5f * clamp(1 - dot(normalize(u_sunDir.xyz), vec3(0, 1, 0)), 0, 1);
    finalColor = mix(finalColor, colorWithFog, fogFactor);

    // set output color
    // -----------------------------------

    fragColor = vec4(finalColor, 1.f);
}
