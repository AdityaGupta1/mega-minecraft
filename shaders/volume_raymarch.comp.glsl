#version 430

#define PI 3.1415926535897932384626f

layout(local_size_x = 320, local_size_y = 1, local_size_z = 1) in;
layout(rgba16f) uniform image3D img_volume;

vec4 accumulateScattering(vec4 colorAndDensityFront, vec4 colorAndDensityBack)
{
    vec3 light = colorAndDensityFront.rgb + clamp(exp(-colorAndDensityFront.a), 0, 1) * colorAndDensityBack.rgb;
    return vec4(light.rgb, colorAndDensityFront.a + colorAndDensityBack.a);
}

void writeOutput(ivec3 volumePos, vec4 colorAndDensity)
{
    vec4 finalValue = vec4(colorAndDensity.rgb, clamp(exp(-colorAndDensity.a), 0, 1));
    imageStore(img_volume, volumePos, finalValue);
}

void main()
{
    ivec3 volumePos = ivec3(gl_GlobalInvocationID.xy, 0);
    vec4 currentSliceValue = imageLoad(img_volume, volumePos);
    writeOutput(volumePos, currentSliceValue);

    for (int z = 1; z < 128; z++)
    {
        ivec3 volumePos = ivec3(gl_GlobalInvocationID.xy, z);
        vec4 nextValue = imageLoad(img_volume, volumePos);
        currentSliceValue = accumulateScattering(currentSliceValue, nextValue);
        writeOutput(volumePos, currentSliceValue);
    }
}