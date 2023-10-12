#include "renderer.hpp"

#include <iostream>
#include "rendering/renderingUtils.hpp"

Renderer::Renderer(GLFWwindow* window, Terrain* terrain)
    : window(window), terrain(terrain), passthroughShader()
{
}

Chunk chunk = Chunk(ivec2(0, 0));

void Renderer::initShaders()
{
    passthroughShader.create("shaders/passthrough.vert.glsl", "shaders/passthrough.frag.glsl");

    chunk.createVBO();
}

void Renderer::init()
{
    glClearColor(1.f, 0.f, 1.f, 1.f);
    glEnable(GL_DEPTH_TEST);

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    initShaders();
}

void Renderer::draw()
{
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, width, height);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    //terrain->draw();

    passthroughShader.draw(chunk);

    glfwSwapBuffers(window);
}