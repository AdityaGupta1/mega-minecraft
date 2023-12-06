#pragma once

// ============================================================
// CONFIG FLAGS
// ============================================================

#define USE_D3D11_RENDERER 1
#define GPU_DEVICE 0

// OptiX
#define USE_DENOISING 0
#define USE_UPSCALING 0
#define RESET_CAMERA_ON_BUILD_IAS 1

// ============================================================
// DEBUG FLAGS
// ============================================================

#define DEBUG_USE_GL_RENDERER 0
#define DEBUG_START_IN_FREE_CAM_MODE 1

// ============================================================
// MATH CONSTANTS
// ============================================================

#define PI              3.14159265358979323846264338327f
#define TWO_PI          6.28318530717958647692528676655f
#define PI_OVER_TWO     1.57079632679489661923132169163f
#define PI_OVER_FOUR    0.78539816339744830961566084581f
#define SQRT_2          1.41421356237309504880168872420f