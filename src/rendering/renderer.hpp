#pragma once

#include "shaderProgram.hpp"
#include <GLFW/glfw3.h>
#include "terrain/terrain.hpp"
#include "player/player.hpp"
#include "fullscreenTri.hpp"

class Renderer
{
private:
    GLFWwindow* window{ nullptr };
    ivec2* windowSize{ nullptr };
    Terrain* terrain{ nullptr };
    Player* player{ nullptr };

    GLuint vao;

    // drawables
    FullscreenTri fullscreenTri;

    // shaders
    ShaderProgram passthroughShader;
    ShaderProgram passthroughUvsShader;
    ShaderProgram lambertShader;

    // textures
    GLuint tex_blockDiffuse;

    // other
    mat4 projMat;
    mat4 viewProjMat;

public:
    Renderer(GLFWwindow* window, ivec2* windowSize, Terrain* terrain, Player* player);

    ~Renderer();

private:
    void initShaders();
    void initTextures();

    void setProjMat();

public:
    void init();

    void draw(bool viewMatChanged, bool windowSizeChanged);
};