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

    GLint unif_modelMat;
    GLint unif_viewProjMat;

    GLint tex_blockDiffuse;

    ShaderProgram();

    bool create(const std::string& vertFile, const std::string& fragFile);

    void useMe() const;

    void setModelMat(const glm::mat4& mat) const;
    void setViewProjMat(const glm::mat4& mat) const;

    void setTexBlockDiffuse(int tex) const;

    void draw(Drawable &d) const;
};