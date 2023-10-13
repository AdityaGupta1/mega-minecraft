#include "renderer.hpp"

#include <iostream>
#include "rendering/renderingUtils.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include "util/utils.hpp"

Renderer::Renderer(GLFWwindow* window, Terrain* terrain, Player* player)
    : window(window), terrain(terrain), player(player), passthroughShader(), lambertShader()
{
}

Chunk chunk = Chunk(ivec2(0, 0));

void Renderer::initShaders()
{
    passthroughShader.create("shaders/passthrough.vert.glsl", "shaders/passthrough.frag.glsl");
    lambertShader.create("shaders/lambert.vert.glsl", "shaders/lambert.frag.glsl");

    chunk.createVBOs();
}

void Renderer::setProjMat()
{
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    projMat = glm::perspective(PI_OVER_FOUR, width / (float)height, 0.01f, 500.f);
}

void Renderer::init()
{
    glClearColor(1.f, 0.f, 1.f, 1.f);
    glEnable(GL_DEPTH_TEST);

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    initShaders();

    setProjMat();
}

void Renderer::draw(bool viewMatChanged)
{
    if (viewMatChanged)
    {
        viewProjMat = projMat * player->getViewMat();
        lambertShader.setViewProjMat(viewProjMat);
    }

    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, width, height);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    //terrain->draw();

    lambertShader.draw(chunk);

    glfwSwapBuffers(window);
}