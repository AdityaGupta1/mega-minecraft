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
    GLuint compShader;
    GLuint prog;

    GLint attr_pos;
    GLint attr_nor;
    GLint attr_uv;

    GLint unif_modelMat;
    GLint unif_viewProjMat;
    GLint unif_invViewProjMat;
    GLint unif_viewMat;
    GLint unif_invViewMat;
    GLint unif_projMat;
    GLint unif_sunViewProjMat;

    GLint unif_sunDir;
    GLint unif_moonDir;
    GLint unif_fogColor;

    GLint tex_blockDiffuse;
    GLint tex_bufColor;
    GLint tex_shadowMap;
    GLint tex_volume; // image or texture, either one

    ShaderProgram();

    void createUniformVariables();

    bool create(const std::string& vertFile, const std::string& fragFile);
    bool createCompute(const std::string& compFile);

    void useMe() const;

    void setModelMat(const glm::mat4& mat) const;
    void setViewProjMat(const glm::mat4& mat) const;
    void setInvViewProjMat(const glm::mat4& mat) const;
    void setViewMat(const glm::mat4& mat) const;
    void setInvViewMat(const glm::mat4& mat) const;
    void setProjMat(const glm::mat4& mat) const;
    void setSunViewProjMat(const glm::mat4& mat) const;

    void setSunDir(const glm::vec4& dir) const;
    void setMoonDir(const glm::vec4& dir) const;
    void setFogColor(const glm::vec3& col) const;

    void setTexBlockDiffuse(int tex) const;
    void setTexBufColor(int tex) const;
    void setTexShadowMap(int tex) const;
    void setTexVolume(int tex) const;

    void draw(Drawable &d) const;
    void dispatchCompute(int groupsX, int groupsY, int groupsZ);
};