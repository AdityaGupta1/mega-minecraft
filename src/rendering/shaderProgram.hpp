#pragma once

#include <GL/glew.h>
#include <glm/glm.hpp>
#include <string>
#include "drawable.hpp"

class ShaderProgram
{
public:
    GLuint vertShader;
    GLuint fragShader;
    GLuint prog;

    int attrPos;
    int attrUv;

    int unifViewProjMat;

    ShaderProgram();

    bool create(const std::string& vertFile, const std::string& fragFile);

    void useMe();

    void setViewProjMat(const glm::mat4& mat);

    void draw(Drawable &d);
};