#include "renderer.hpp"

#include <iostream>

Renderer::Renderer()
    : passthroughShader()
{
}

void Renderer::initShaders()
{
    passthroughShader.create("shaders/passthrough.vert.glsl", "shaders/passthrough.frag.glsl");
}