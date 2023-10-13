#include "shaderProgram.hpp"

#include "util/utils.hpp"
#include <iostream>
#include "renderingUtils.hpp"
#include "structs.hpp"

ShaderProgram::ShaderProgram()
    : vertShader(), fragShader(), prog(), attrPos(-1), unifViewProjMat(-1)
{
}

bool ShaderProgram::create(const std::string& vertFile, const std::string& fragFile)
{
    std::cout << "creating shader from " << vertFile << " and " << fragFile << "..." << std::endl;

    vertShader = glCreateShader(GL_VERTEX_SHADER);
    fragShader = glCreateShader(GL_FRAGMENT_SHADER);
    prog = glCreateProgram();

    const std::string vertText = Utils::readFile(vertFile);
    const std::string fragText = Utils::readFile(fragFile);

    const char* vertChars = vertText.c_str();
    const char* fragChars = fragText.c_str();

    glShaderSource(vertShader, 1, (const char**)&vertChars, 0);
    glShaderSource(fragShader, 1, (const char**)&fragChars, 0);
    glCompileShader(vertShader);
    glCompileShader(fragShader);

    GLint compiled;
    glGetShaderiv(vertShader, GL_COMPILE_STATUS, &compiled);
    if (!compiled)
    {
        printf("vertex shader didn't compile\n"); // TODO print actual error message
        return false;
    }
    glGetShaderiv(fragShader, GL_COMPILE_STATUS, &compiled);
    if (!compiled)
    {
        printf("fragment shader didn't compile\n"); // TODO print actual error message
        return false;
    }

    glAttachShader(prog, vertShader);
    glAttachShader(prog, fragShader);
    glLinkProgram(prog);

    GLint linked;
    glGetProgramiv(prog, GL_LINK_STATUS, &linked);
    if (!linked)
    {
        printf("shader didn't link\n"); // TODO print actual error message
        return false;
    }

    attrPos = glGetAttribLocation(prog, "v_pos");

    unifViewProjMat = glGetUniformLocation(prog, "u_viewProjMat");

    std::cout << "done" << std::endl;
    return true;
}

void ShaderProgram::useMe()
{
    glUseProgram(prog);
}

void ShaderProgram::setViewProjMat(const glm::mat4& mat)
{
    useMe();
    glUniformMatrix4fv(unifViewProjMat, 1, GL_FALSE, &mat[0][0]);
}

void ShaderProgram::draw(Drawable& d)
{
    useMe();

    if (d.getIdxCount() < 0)
    {
        throw std::out_of_range("drawable has no elements");
    }

    if (d.bindVerts())
    {
        if (attrPos != -1)
        {
            glEnableVertexAttribArray(attrPos);
            glVertexAttribPointer(attrPos, 3, GL_FLOAT, false, sizeof(Vertex), (void*)0);
        }
    }

    d.bindIdx();
    glDrawElements(d.drawMode(), d.getIdxCount(), GL_UNSIGNED_INT, 0);

    // TODO disable vertex attrib arrays here? I don't think that's necessary though

    RenderingUtils::printGLErrors();
}