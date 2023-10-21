#version 330

uniform mat4 u_viewTransposeMat;
uniform mat4 u_projMat;

uniform vec3 u_sunDir;

in vec2 fs_uv;

out vec4 fragColor;

void main() {
    vec2 ndc = fs_uv * 2.f - 1.f;
    vec3 camForward = -vec3(u_viewTransposeMat[2]); // not sure why this is backwards but whatever
    vec3 camRight = vec3(u_viewTransposeMat[0]);
    vec3 camUp = vec3(u_viewTransposeMat[1]);
    vec3 worldDir = normalize(camForward + (ndc.x / u_projMat[0][0] * camRight) + (ndc.y / u_projMat[1][1] * camUp));

    if (dot(worldDir, u_sunDir) > 0.998f) {
        fragColor = vec4(1.f, 1.f, 1.f, 1.f);
        return;
    }

    fragColor = vec4(worldDir, 1.f);
}
