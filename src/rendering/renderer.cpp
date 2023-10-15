#include "renderer.hpp"

#include <iostream>
#include "rendering/renderingUtils.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include "util/utils.hpp"

// TODO temporary includes for testing
#include "cuda/cuda_utils.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <iostream>

Renderer::Renderer(GLFWwindow* window, ivec2* windowSize, Terrain* terrain, Player* player)
    : window(window), windowSize(windowSize), terrain(terrain), player(player),
      passthroughShader(), lambertShader()
{
}

Chunk chunk = Chunk(ivec2(0, 0));

void Renderer::initShaders()
{
    passthroughShader.create("shaders/passthrough.vert.glsl", "shaders/passthrough.frag.glsl");
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

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    initShaders();
    initTextures();

    setProjMat();

    // TODO: rest of this function is temporary testing code

    Block* dev_blocks;
    unsigned char* dev_heightfield;

    cudaMalloc((void**)&dev_blocks, 65536 * sizeof(Block));
    cudaMalloc((void**)&dev_heightfield, 256 * sizeof(unsigned char));

    CudaUtils::checkCUDAError("cudaMalloc failed");

    chunk.dummyFillCUDA(dev_blocks, dev_heightfield);
    chunk.createVBOs();
    chunk.bufferVBOs();

    cudaFree(dev_blocks);
    cudaFree(dev_heightfield);

    CudaUtils::checkCUDAError("cudaFree failed");
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

    //terrain->draw();

    lambertShader.draw(chunk);

    glfwSwapBuffers(window);
}