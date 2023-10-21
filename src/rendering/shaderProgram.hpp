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
    GLint attr_nor;
    GLint attr_uv;

    GLint unif_modelMat;
    GLint unif_viewProjMat;
    GLint unif_invViewProjMat; // TODO: see if this actually ends up getting used somewhere
    GLint unif_viewTransposeMat;
    GLint unif_projMat;
    GLint unif_sunViewProjMat;

    GLint unif_sunDir;

    GLint tex_blockDiffuse;
    GLint tex_bufColor;
    GLint tex_shadowMap;

    ShaderProgram();

    bool create(const std::string& vertFile, const std::string& fragFile);

    void useMe() const;

    void setModelMat(const glm::mat4& mat) const;
    void setViewProjMat(const glm::mat4& mat) const;
    void setInvViewProjMat(const glm::mat4& mat) const;
    void setViewTransposeMat(const glm::mat4& mat) const;
    void setProjMat(const glm::mat4& mat) const;
    void setSunViewProjMat(const glm::mat4& mat) const;

    void setSunDir(const glm::vec3& dir) const;

    void setTexBlockDiffuse(int tex) const;
    void setTexBufColor(int tex) const;
    void setTexShadowMap(int tex) const;

    void draw(Drawable &d) const;
};