#include "renderer.hpp"

#include <iostream>
#include "rendering/renderingUtils.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include "util/utils.hpp"
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define SHADOW_MAP_SIZE 8192

#define DEBUG_DISABLE_RENDERING 0

#define DEBUG_CREATE_PASSTHROUGH_SHADERS 0
#define DEBUG_DISPLAY_SHADOW_MAP 0

Renderer::Renderer(GLFWwindow* window, ivec2* windowSize, Terrain* terrain, Player* player)
    : window(window), windowSize(windowSize), terrain(terrain), player(player), vao(-1),
      fbo_main(-1), rbo_main(-1), fbo_shadow(-1), fbo_postprocess1(-1), fbo_bloom1(-1), fbo_bloom2(-1),
      tex_blockDiffuse(-1), tex_bufColor1(-1), tex_bufColor2(-1), tex_shadow(-1), tex_volume(-1), tex_bufBloomColor1(-1), tex_bufBloomColor2(-1)
{
    float orthoSize = 420.f;
    sunProjMat = glm::ortho<float>(
        -orthoSize, orthoSize,
        -orthoSize, orthoSize,
        -1000.f, 1000.f
    );
}

Renderer::~Renderer()
{
    glDeleteVertexArrays(1, &vao);
}

bool Renderer::init()
{
    glClearColor(0.37f, 1.f, 0.94f, 1.f);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    fullscreenTri.bufferVBOs();

    if (!initShaders())
    {
        return false;
    }

    if (!initFbosAndTextures())
    {
        return false;
    }

    setProjMat();

    const vec3 sunAxisForward = normalize(vec3(6.0f, -2.0f, 2.0f));
    const vec3 sunAxisRight = normalize(cross(sunAxisForward, vec3(0, 1, 0)));
    const vec3 sunAxisUp = normalize(cross(sunAxisRight, sunAxisForward));
    sunRotateMat = mat3(sunAxisRight, sunAxisForward, sunAxisUp);

    return true;
}

bool createCustomShader(ShaderProgram& prog, const std::string& name)
{
    return prog.create("shaders/" + name + ".vert.glsl", "shaders/" + name + ".frag.glsl");
}

bool createPostProcessShader(ShaderProgram& prog, const std::string& frag)
{
    return prog.create("shaders/passthrough_uvs.vert.glsl", "shaders/" + frag + ".frag.glsl");
}

bool createComputeShader(ShaderProgram& prog, const std::string& comp)
{
    return prog.createCompute("shaders/" + comp + ".comp.glsl");
}

bool Renderer::initShaders()
{
    bool success = true;

#if DEBUG_CREATE_PASSTHROUGH_SHADERS
    success &= passthroughShader.create("shaders/passthrough.vert.glsl", "shaders/passthrough.frag.glsl");
    success &= passthroughUvsShader.create("shaders/passthrough_uvs.vert.glsl", "shaders/passthrough_uvs.frag.glsl");
#endif

    success &= createCustomShader(lambertShader, "lambert");
    success &= createPostProcessShader(skyShader, "sky");
    success &= createCustomShader(shadowShader, "shadow");
    success &= createPostProcessShader(postProcessShader1, "postprocess_1");
    success &= createPostProcessShader(postProcessShaderFinal, "postprocess_final");
    success &= createPostProcessShader(bloomBlurShader, "bloom_blur");

    success &= createComputeShader(volumeFillShader, "volume_fill");
    success &= createComputeShader(volumeRaymarchShader, "volume_raymarch");

    if (RenderingUtils::printGLErrors())
    {
        std::cerr << "ERROR: Renderer::initShaders()" << std::endl;
        return false;
    }

    return success;
}

bool checkFramebufferStatus(const std::string& name)
{
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    {
        std::cout << name << " framebuffer no worky" << std::endl;
        return false;
    }

    return true;
}

bool Renderer::initFbosAndTextures()
{
    // ============================================================
    // BLOCK TEXTURES
    // ============================================================

    stbi_set_flip_vertically_on_load(true);

    glActiveTexture(GL_TEXTURE0);
    glGenTextures(1, &tex_blockDiffuse);
    glBindTexture(GL_TEXTURE_2D, tex_blockDiffuse);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    int width, height, channels;
    unsigned char* data = stbi_load("textures/blocks_diffuse.png", &width, &height, &channels, 0);

    for (int i = 0; i < width * height; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            auto& col = data[i * 4 + j];
            col = (unsigned char)(pow(col / 255.f, 2.2f) * 255.f);
        }
    }

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    stbi_image_free(data);

    lambertShader.setTexBlockDiffuse(0);
    shadowShader.setTexBlockDiffuse(0); // for testing alpha of fragments

    std::cout << "loaded blocks_diffuse.png" << std::endl;

    // ============================================================
    // MAIN FRAMEBUFFER
    // ============================================================

    glGenFramebuffers(1, &fbo_main);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_main);

    glActiveTexture(GL_TEXTURE1);
    glGenTextures(1, &tex_bufColor1);
    glBindTexture(GL_TEXTURE_2D, tex_bufColor1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, windowSize->x, windowSize->y, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_bufColor1, 0);

    glGenRenderbuffers(1, &rbo_main);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo_main);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, windowSize->x, windowSize->y);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo_main);

    if (!checkFramebufferStatus("main"))
    {
        return false;
    }

    postProcessShader1.setTexBufColor(1);

    std::cout << "created fbo_main" << std::endl;

    // ============================================================
    // SHADOW FRAMEBUFFER
    // ============================================================

    glGenFramebuffers(1, &fbo_shadow);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_shadow);

    glActiveTexture(GL_TEXTURE2);
    glGenTextures(1, &tex_shadow);
    glBindTexture(GL_TEXTURE_2D, tex_shadow);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // set shadow depth = 1.0 for coords outside texture so those coords will never be in shadow
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    float borderColor[] = { 1.0f, 1.0f, 1.0f, 1.0f };
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE);

    glBindFramebuffer(GL_FRAMEBUFFER, fbo_shadow);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, tex_shadow, 0);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);

    if (!checkFramebufferStatus("shadow"))
    {
        return false;
    }

    lambertShader.setTexShadowMap(2);
    volumeFillShader.setTexShadowMap(2);

#if DEBUG_DISPLAY_SHADOW_MAP
    postProcessShader1.setTexBufColor(2);
#endif

    std::cout << "created fbo_shadow" << std::endl;

    // ============================================================
    // VOLUMETRIC FOG TEXTURE
    // ============================================================

    glGenTextures(1, &tex_volume);
    // 3 is the TEXTURE IMAGE UNIT
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_3D, tex_volume);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA16F, 320, 180, 128, 0, GL_RGBA, GL_FLOAT, 0);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    // 0 is the IMAGE UNIT
    glBindImageTexture(0, tex_volume, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA16F);

    lambertShader.setTexVolume(3);
    skyShader.setTexVolume(3);

    volumeFillShader.setTexVolume(0);
    volumeRaymarchShader.setTexVolume(0);

    std::cout << "created tex_volume" << std::endl;

    // ============================================================
    // POSTPROCESS FRAMEBUFFER 1
    // ============================================================

    glGenFramebuffers(1, &fbo_postprocess1);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_postprocess1);

    glActiveTexture(GL_TEXTURE4);
    glGenTextures(1, &tex_bufColor2);
    glBindTexture(GL_TEXTURE_2D, tex_bufColor2);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, windowSize->x, windowSize->y, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_bufColor2, 0);

    glActiveTexture(GL_TEXTURE5);
    glGenTextures(1, &tex_bufBloomColor2);
    glBindTexture(GL_TEXTURE_2D, tex_bufBloomColor2);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, windowSize->x, windowSize->y, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glGenTextures(1, &tex_bufBloomColor1);
    glBindTexture(GL_TEXTURE_2D, tex_bufBloomColor1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, windowSize->x, windowSize->y, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, tex_bufBloomColor1, 0);

    GLuint attachments[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
    glDrawBuffers(2, attachments);

    if (!checkFramebufferStatus("postprocess 1"))
    {
        return false;
    }

    postProcessShaderFinal.setTexBufColor(4);
    postProcessShaderFinal.setTexBufBloomColor(5);

    bloomBlurShader.setTexBufBloomColor(5);

    std::cout << "created fbo_postprocess1" << std::endl;

    // ============================================================
    // BLOOM
    // ============================================================

    glGenFramebuffers(1, &fbo_bloom1);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_bloom1);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_bufBloomColor2, 0);

    if (!checkFramebufferStatus("bloom 1"))
    {
        return false;
    }

    glGenFramebuffers(1, &fbo_bloom2);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_bloom2);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_bufBloomColor1, 0);

    if (!checkFramebufferStatus("bloom 2"))
    {
        return false;
    }

    std::cout << "created fbo_bloom1 and fbo_bloom2" << std::endl;

    if (RenderingUtils::printGLErrors())
    {
        std::cerr << "ERROR: Renderer::initFbosAndTextures()" << std::endl;
        return false;
    }

    return true;
}

void Renderer::resizeTextures()
{
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, tex_bufColor1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, windowSize->x, windowSize->y, 0, GL_RGBA, GL_FLOAT, NULL);

    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_2D, tex_bufColor2);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, windowSize->x, windowSize->y, 0, GL_RGBA, GL_FLOAT, NULL);
    
    glActiveTexture(GL_TEXTURE5);
    glBindTexture(GL_TEXTURE_2D, tex_bufBloomColor2);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, windowSize->x, windowSize->y, 0, GL_RGBA, GL_FLOAT, NULL);
    glBindTexture(GL_TEXTURE_2D, tex_bufBloomColor1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, windowSize->x, windowSize->y, 0, GL_RGBA, GL_FLOAT, NULL);

    glBindRenderbuffer(GL_RENDERBUFFER, rbo_main);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, windowSize->x, windowSize->y);
}

static constexpr float fovNormal = glm::radians(52.f);
static constexpr float fovZoomed = glm::radians(20.f);

void Renderer::setProjMat()
{
    float fovy = this->isZoomed ? fovZoomed : fovNormal;
    projMat = glm::perspective(fovy, windowSize->x / (float)windowSize->y, 0.05f, 5000.f);
}

void Renderer::setZoomed(bool zoomed)
{
    this->isZoomed = zoomed;
    setProjMat();
}

void Renderer::toggleTimePaused()
{
    this->isTimePaused = !this->isTimePaused;
}

vec4 sunDir3To4(const vec3& sunDirXYZ)
{
    return vec4(sunDirXYZ, smoothstep(-0.1f, 0.1f, sunDirXYZ.y));
}

void Renderer::draw(float deltaTime, bool viewMatChanged, bool windowSizeChanged)
{
#if DEBUG_DISABLE_RENDERING
    return;
#endif

    if (windowSizeChanged)
    {
        setProjMat();
        resizeTextures();
    }

    bool setViewMat = viewMatChanged || windowSizeChanged;
    if (setViewMat)
    {
        viewMat = player->getViewMat();
        viewProjMat = projMat * viewMat;
        invViewProjMat = glm::inverse(viewProjMat);
        invViewMat = glm::inverse(player->getViewMat());
    }

    if (!isTimePaused)
    {
        time += deltaTime;
    }

    const float sunTime = time * 0.2f + 0.4f;
    float cosSunTime = cos(sunTime);
    float sinSunTime = sin(sunTime);
    const vec4 sunDir = sunDir3To4(normalize(sunRotateMat * vec3(cosSunTime, 0.55f, sinSunTime)));
    const vec4 moonDir = sunDir3To4(normalize(sunRotateMat * vec3(-cosSunTime, 0.55f, -sinSunTime)));

    // ============================================================
    // SHADOW
    // ============================================================

    glBindFramebuffer(GL_FRAMEBUFFER, fbo_shadow);
    glViewport(0, 0, SHADOW_MAP_SIZE, SHADOW_MAP_SIZE);
    glClear(GL_DEPTH_BUFFER_BIT);

    glEnable(GL_DEPTH_TEST);
    glCullFace(GL_FRONT);

    const auto terrainCurrentChunkPos = terrain->getCurrentChunkPos();
    const vec3 sunMoonViewCenterPos = vec3(terrainCurrentChunkPos.x * 16, 0, terrainCurrentChunkPos.y * 16);
    vec3 sunMoonDir = sunDir.w > 0 ? vec3(sunDir) : vec3(moonDir);
    const mat4 sunViewMat = glm::lookAt(sunMoonViewCenterPos + sunMoonDir, sunMoonViewCenterPos, vec3(0, 1, 0));
    const mat4 sunViewProjMat = sunProjMat * sunViewMat;

    shadowShader.setSunViewProjMat(sunViewProjMat);
    terrain->draw(shadowShader, nullptr);

    // ============================================================
    // VOLUMETRIC FOG
    // ============================================================

    volumeFillShader.setSunViewProjMat(sunViewProjMat);
    volumeFillShader.setFogColor(sunDir.w * vec3(1.0f, 1.0f, 0.93f) + moonDir.w * vec3(0.93f, 0.98f, 1.0f) * 0.22f);

    if (setViewMat)
    {
        volumeFillShader.setViewProjMat(viewProjMat);
        volumeFillShader.setInvViewProjMat(invViewProjMat);
        volumeFillShader.setInvViewMat(invViewMat);
        volumeFillShader.setProjMat(projMat);
    }

    volumeFillShader.dispatchCompute(1, 180, 128);
    glMemoryBarrier(GL_ALL_BARRIER_BITS);

    volumeRaymarchShader.dispatchCompute(1, 180, 1);
    glMemoryBarrier(GL_ALL_BARRIER_BITS);

    // ============================================================
    // G-BUFFER (MAIN)
    // ============================================================

    glBindFramebuffer(GL_FRAMEBUFFER, fbo_main);
    glViewport(0, 0, windowSize->x, windowSize->y);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glCullFace(GL_BACK);

    if (setViewMat)
    {
        lambertShader.setViewProjMat(viewProjMat);
    }

    lambertShader.setSunViewProjMat(sunViewProjMat);
    lambertShader.setSunDir(sunDir);
    lambertShader.setMoonDir(moonDir);

    terrain->draw(lambertShader, player);

    if (setViewMat)
    {
        skyShader.setInvViewMat(invViewMat);
        skyShader.setProjMat(projMat);
    }

    skyShader.setSunDir(sunDir);
    skyShader.setMoonDir(moonDir);

    skyShader.draw(fullscreenTri);

    // ============================================================
    // POSTPROCESS 1
    // ============================================================

    glBindFramebuffer(GL_FRAMEBUFFER, fbo_postprocess1);

    glDisable(GL_DEPTH_TEST);

    postProcessShader1.draw(fullscreenTri);

    // ============================================================
    // BLOOM
    // ============================================================

    glActiveTexture(GL_TEXTURE5);
    bool horizontal = true;
    int amount = 5;
    for (int i = 0; i < amount * 2; ++i)
    {
        glBindFramebuffer(GL_FRAMEBUFFER, horizontal ? fbo_bloom1 : fbo_bloom2);
        bloomBlurShader.setHorizontal(horizontal);
        glBindTexture(GL_TEXTURE_2D, horizontal ? tex_bufBloomColor1 : tex_bufBloomColor2);
        bloomBlurShader.draw(fullscreenTri);
        horizontal = !horizontal;
    }
    glBindTexture(GL_TEXTURE_2D, tex_bufBloomColor1);

    // ============================================================
    // VIEWPORT (POSTPROCESS FINAL)
    // ============================================================

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    postProcessShaderFinal.draw(fullscreenTri);

    glfwSwapBuffers(window);
}