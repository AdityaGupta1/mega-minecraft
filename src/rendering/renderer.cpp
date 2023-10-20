#include "renderer.hpp"

#include <iostream>
#include "rendering/renderingUtils.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include "util/utils.hpp"
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define CREATE_PASSTHROUGH_SHADERS 0

Renderer::Renderer(GLFWwindow* window, ivec2* windowSize, Terrain* terrain, Player* player)
    : window(window), windowSize(windowSize), terrain(terrain), player(player), vao(-1),
      fbo_main(-1), tex_blockDiffuse(-1), projMat(), viewProjMat(), invViewProjMat(), sunRotateMat()
{}

Renderer::~Renderer()
{
    glDeleteVertexArrays(1, &vao);
}

void Renderer::init()
{
    glClearColor(0.37f, 1.f, 0.94f, 1.f);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    fullscreenTri.bufferVBOs();

    initShaders();
    initFbosAndTextures();

    setProjMat();

    const vec3 sunAxisForward = normalize(vec3(5, 2, 3));
    const vec3 sunAxisRight = normalize(cross(sunAxisForward, vec3(0, 1, 0)));
    const vec3 sunAxisUp = normalize(cross(sunAxisRight, sunAxisForward));
    sunRotateMat = mat3(sunAxisRight, sunAxisForward, sunAxisUp);
}

void createPostProcessShader(ShaderProgram& prog, const std::string& frag)
{
    prog.create("shaders/passthrough_uvs.vert.glsl", frag);
}

void Renderer::initShaders()
{
#if CREATE_PASSTHROUGH_SHADERS
    passthroughShader.create("shaders/passthrough.vert.glsl", "shaders/passthrough.frag.glsl");
    passthroughUvsShader.create("shaders/passthrough_uvs.vert.glsl", "shaders/passthrough_uvs.frag.glsl");
#endif

    lambertShader.create("shaders/lambert.vert.glsl", "shaders/lambert.frag.glsl");
    createPostProcessShader(skyShader, "shaders/sky.frag.glsl");
    createPostProcessShader(postProcessShader1, "shaders/postprocess_1.frag.glsl");
}

void Renderer::initFbosAndTextures()
{
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

    std::cout << "loaded blocks_diffuse.png" << std::endl;

    glGenFramebuffers(1, &fbo_main);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_main);

    glActiveTexture(GL_TEXTURE1);
    glGenTextures(1, &tex_bufColor);
    glBindTexture(GL_TEXTURE_2D, tex_bufColor);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, windowSize->x, windowSize->y, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_bufColor, 0);

    glGenRenderbuffers(1, &rbo_main);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo_main);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, windowSize->x, windowSize->y);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo_main);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    {
        std::cout << "framebuffer no worky" << std::endl;
        return;
    }

    postProcessShader1.setTexBufColor(1);

    std::cout << "created fbo_main" << std::endl;
}

void Renderer::resizeTextures()
{
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, tex_bufColor);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, windowSize->x, windowSize->y, 0, GL_RGBA, GL_FLOAT, NULL);

    glBindRenderbuffer(GL_RENDERBUFFER, rbo_main);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, windowSize->x, windowSize->y);
}

void Renderer::setProjMat()
{
    float fovy = this->isZoomed ? (PI_OVER_FOUR / 2.8f) : PI_OVER_FOUR;
    projMat = glm::perspective(fovy, windowSize->x / (float)windowSize->y, 0.01f, 1000.f);
}

void Renderer::setZoomed(bool zoomed)
{
    this->isZoomed = zoomed;
    setProjMat();
}

void Renderer::draw(float deltaTime, bool viewMatChanged, bool windowSizeChanged)
{
    if (windowSizeChanged)
    {
        setProjMat();
        resizeTextures();
    }

    if (viewMatChanged || windowSizeChanged)
    {
        viewProjMat = projMat * player->getViewMat();
        invViewProjMat = glm::inverse(viewProjMat);

        lambertShader.setViewProjMat(viewProjMat);

        skyShader.setInvViewProjMat(invViewProjMat);
    }

    time += deltaTime;

    const float sunTime = time * 0.2f;
    const vec3 sunDir = normalize(sunRotateMat * vec3(cos(sunTime), 0.7f, sin(sunTime)));

    glBindFramebuffer(GL_FRAMEBUFFER, fbo_main);
    glViewport(0, 0, windowSize->x, windowSize->y);
    glEnable(GL_DEPTH_TEST);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    //glActiveTexture(GL_TEXTURE0);
    //glBindTexture(GL_TEXTURE_2D, tex_blockDiffuse);

    lambertShader.setSunDir(sunDir);
    terrain->draw(lambertShader, *player);

    skyShader.setSunDir(sunDir);
    skyShader.draw(fullscreenTri);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, windowSize->x, windowSize->y);
    glDisable(GL_DEPTH_TEST);

    //glActiveTexture(GL_TEXTURE1);
    //glBindTexture(GL_TEXTURE_2D, tex_bufColor);

    postProcessShader1.draw(fullscreenTri);

    glfwSwapBuffers(window);
}