#pragma once

#include "shaderProgram.hpp"

class Renderer
{
public:
    ShaderProgram passthroughShader;

    Renderer();

    void initShaders();
};