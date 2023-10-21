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
    ShaderProgram skyShader;
    ShaderProgram shadowShader;
    ShaderProgram postProcessShader1;

    // framebuffer objects
    GLuint fbo_main, rbo_main;
    GLuint fbo_shadow;

    // textures
    GLuint tex_blockDiffuse;
    GLuint tex_bufColor;
    GLuint tex_shadow;

    // other
    mat4 projMat;
    mat4 viewProjMat;
    mat4 invViewProjMat;

    bool isZoomed{ false };
    bool isTimePaused{ true };

    float time{ 0 };
    mat3 sunRotateMat;

public:
    Renderer(GLFWwindow* window, ivec2* windowSize, Terrain* terrain, Player* player);

    ~Renderer();

private:
    bool initShaders();
    bool initFbosAndTextures();
    void resizeTextures();

    void setProjMat();

public:
    bool init();

    void setZoomed(bool zoomed);
    void toggleTimePaused();

    void draw(float deltaTime, bool viewMatChanged, bool windowSizeChanged);
};