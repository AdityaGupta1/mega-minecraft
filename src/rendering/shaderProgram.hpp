#pragma once

#include <GL/glew.h>
#include <string>
#include "drawable.hpp"

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

    void draw(Drawable &d);
};