#pragma once

#include "shaderProgram.hpp"
#include <GLFW/glfw3.h>
#include "terrain/terrain.hpp"

class Renderer
{
private:
    GLFWwindow* window;
    Terrain* terrain;

    ShaderProgram passthroughShader;

    GLuint vao;

public:
    Renderer(GLFWwindow* window, Terrain* terrain);

private:
    void initShaders();

public:
    void init();

    void draw();
};