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
      tex_blockDiffuse(-1), projMat(), viewProjMat()
{}

Renderer::~Renderer()
{
    fullscreenTri.destroyVBOs();

    glDeleteVertexArrays(1, &vao);
}

void Renderer::initShaders()
{
#if CREATE_PASSTHROUGH_SHADERS
    passthroughShader.create("shaders/passthrough.vert.glsl", "shaders/passthrough.frag.glsl");
    passthroughUvsShader.create("shaders/passthrough_uvs.vert.glsl", "shaders/passthrough_uvs.frag.glsl");
#endif
    lambertShader.create("shaders/lambert.vert.glsl", "shaders/lambert.frag.glsl");
}

void Renderer::initTextures()
{
    stbi_set_flip_vertically_on_load(true);

    glGenTextures(1, &tex_blockDiffuse);
    glBindTexture(GL_TEXTURE_2D, tex_blockDiffuse);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    int width, height, channels;
    unsigned char* data = stbi_load("textures/blocks_diffuse.png", &width, &height, &channels, 0);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    stbi_image_free(data);

    std::cout << "loaded blocks_diffuse.png" << std::endl;

    lambertShader.setTexBlockDiffuse(0);
}

void Renderer::setProjMat()
{
    projMat = glm::perspective(PI_OVER_FOUR, windowSize->x / (float)windowSize->y, 0.01f, 500.f);
}

void Renderer::init()
{
    glClearColor(0.64f, 1.f, 0.97f, 1.f);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    fullscreenTri.bufferVBOs();

    initShaders();
    initTextures();

    setProjMat();
}

void Renderer::draw(bool viewMatChanged, bool windowSizeChanged)
{
    if (windowSizeChanged)
    {
        setProjMat();
    }

    if (viewMatChanged || windowSizeChanged)
    {
        viewProjMat = projMat * player->getViewMat();

        lambertShader.setViewProjMat(viewProjMat);
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, windowSize->x, windowSize->y);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_blockDiffuse);

    terrain->draw(lambertShader, *player);

    glfwSwapBuffers(window);
}