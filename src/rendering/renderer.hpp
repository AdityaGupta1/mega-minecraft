#pragma once

#include "shaderProgram.hpp"
#include <GLFW/glfw3.h>
#include "terrain/terrain.hpp"
#include "player/player.hpp"

class Renderer
{
private:
    GLFWwindow* window{ nullptr };
    Terrain* terrain{ nullptr };
    Player* player{ nullptr };

    GLuint vao;

    ShaderProgram passthroughShader;
    ShaderProgram lambertShader;

    mat4 projMat;
    mat4 viewProjMat;

public:
    Renderer(GLFWwindow* window, Terrain* terrain, Player* player);

private:
    void initShaders();

    void setProjMat();

public:
    void init();

    void draw(bool viewMatChanged);
};