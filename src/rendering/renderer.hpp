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
    ShaderProgram postProcessShader1;

    // framebuffer objects
    GLuint fbo_main;
    GLuint rbo_main;

    // textures
    GLuint tex_blockDiffuse;
    GLuint tex_bufColor;

    // other
    mat4 projMat;
    mat4 viewProjMat;

    bool isZoomed{ false };

public:
    Renderer(GLFWwindow* window, ivec2* windowSize, Terrain* terrain, Player* player);

    ~Renderer();

private:
    void initShaders();
    void initFbosAndTextures();
    void resizeTextures();

    void setProjMat();

public:
    void init();

    void setZoomed(bool zoomed);

    void draw(bool viewMatChanged, bool windowSizeChanged);
};