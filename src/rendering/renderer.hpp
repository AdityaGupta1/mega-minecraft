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

    ShaderProgram volumeFillShader;
    ShaderProgram volumeRaymarchShader;

    // framebuffer objects
    GLuint fbo_main, rbo_main;
    GLuint fbo_shadow;

    // textures
    GLuint tex_blockDiffuse; // tex unit 0
    GLuint tex_bufColor; // tex unit 1
    GLuint tex_shadow; // tex unit 2
    GLuint tex_volume; // tex unit 3, image unit 0

    // other
    mat4 projMat{};
    mat4 viewProjMat{};
    mat4 invViewProjMat{};
    mat4 viewMat{};
    mat4 invViewMat{};
    mat4 sunProjMat{};

    bool isZoomed{ false };
    bool isTimePaused{ true };

    float time{ 0 };
    mat3 sunRotateMat{};

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