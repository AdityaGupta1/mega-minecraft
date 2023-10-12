#pragma once

#include <GL/glew.h>
#include <string>

class ShaderProgram
{
public:
    GLuint vertShader;
    GLuint fragShader;
    GLuint prog;

    int attrPos;

    ShaderProgram();

    bool create(const std::string& vertFile, const std::string& fragFile);

    void useMe();
};