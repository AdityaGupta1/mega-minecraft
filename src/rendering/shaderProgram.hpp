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

    GLint attr_pos;
    GLint attr_uv;

    GLint unif_viewProjMat;

    GLint tex_blockDiffuse;

    ShaderProgram();

    bool create(const std::string& vertFile, const std::string& fragFile);

    void useMe();

    void setViewProjMat(const glm::mat4& mat);

    void setTexBlockDiffuse(int tex);

    void draw(Drawable &d);
};