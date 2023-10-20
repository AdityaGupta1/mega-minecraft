#include "shaderProgram.hpp"

#include "util/utils.hpp"
#include <iostream>
#include "renderingUtils.hpp"
#include "structs.hpp"

ShaderProgram::ShaderProgram()
    : vertShader(), fragShader(), prog(), attr_pos(-1), attr_nor(-1), attr_uv(-1), unif_modelMat(-1), unif_viewProjMat(-1),
      tex_blockDiffuse(-1), tex_bufColor(-1)
{}

void printShaderInfoLog(int shader)
{
    int infoLogLen = 0;
    int charsWritten = 0;
    GLchar* infoLog;

    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLogLen);

    if (infoLogLen > 0)
    {
        infoLog = new GLchar[infoLogLen];
        glGetShaderInfoLog(shader, infoLogLen, &charsWritten, infoLog);
        std::cerr << "ShaderInfoLog:" << "\n" << infoLog << "\n";
        delete[] infoLog;
    }
}

void printLinkInfoLog(int prog)
{
    int infoLogLen = 0;
    int charsWritten = 0;
    GLchar* infoLog;

    glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &infoLogLen);

    if (infoLogLen > 0)
    {
        infoLog = new GLchar[infoLogLen];
        // error check for fail to allocate memory omitted
        glGetProgramInfoLog(prog, infoLogLen, &charsWritten, infoLog);
        std::cerr << "LinkInfoLog:" << "\n" << infoLog << "\n";
        delete[] infoLog;
    }
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
        printShaderInfoLog(vertShader);
        return false;
    }
    glGetShaderiv(fragShader, GL_COMPILE_STATUS, &compiled);
    if (!compiled)
    {
        printShaderInfoLog(fragShader);
        return false;
    }

    glAttachShader(prog, vertShader);
    glAttachShader(prog, fragShader);
    glLinkProgram(prog);

    GLint linked;
    glGetProgramiv(prog, GL_LINK_STATUS, &linked);
    if (!linked)
    {
        printLinkInfoLog(prog);
        return false;
    }

    attr_pos = glGetAttribLocation(prog, "vs_pos");
    attr_nor = glGetAttribLocation(prog, "vs_nor");
    attr_uv = glGetAttribLocation(prog, "vs_uv");

    unif_modelMat = glGetUniformLocation(prog, "u_modelMat");
    unif_viewProjMat = glGetUniformLocation(prog, "u_viewProjMat");

    tex_blockDiffuse = glGetUniformLocation(prog, "tex_blockDiffuse");
    tex_bufColor = glGetUniformLocation(prog, "tex_bufColor");

    std::cout << "done" << std::endl;
    return true;
}

void ShaderProgram::useMe() const
{
    glUseProgram(prog);
}

void ShaderProgram::setModelMat(const glm::mat4& mat) const
{
    useMe();
    glUniformMatrix4fv(unif_modelMat, 1, GL_FALSE, &mat[0][0]);
}

void ShaderProgram::setViewProjMat(const glm::mat4& mat) const
{
    useMe();
    glUniformMatrix4fv(unif_viewProjMat, 1, GL_FALSE, &mat[0][0]);
}

void ShaderProgram::setTexBlockDiffuse(int tex) const
{
    useMe();
    glUniform1i(tex_blockDiffuse, tex);
}

void ShaderProgram::setTexBufColor(int tex) const
{
    useMe();
    glUniform1i(tex_bufColor, tex);
}

void ShaderProgram::draw(Drawable& d) const
{
    useMe();

    if (d.getIdxCount() < 0)
    {
        throw std::out_of_range("drawable has no elements");
    }

    if (d.bindVerts())
    {
        if (attr_pos != -1)
        {
            glEnableVertexAttribArray(attr_pos);
            glVertexAttribPointer(attr_pos, 3, GL_FLOAT, false, sizeof(Vertex), (void*)0);
        }

        if (attr_nor != -1)
        {
            glEnableVertexAttribArray(attr_nor);
            glVertexAttribPointer(attr_nor, 3, GL_FLOAT, false, sizeof(Vertex), (void*)(sizeof(glm::vec3)));
        }

        if (attr_uv != -1)
        {
            glEnableVertexAttribArray(attr_uv);
            glVertexAttribPointer(attr_uv, 2, GL_FLOAT, false, sizeof(Vertex), (void*)(2 * sizeof(glm::vec3)));
        }
    }
    else if (d.bindTriInfo())
    {
        if (attr_pos != -1)
        {
            glEnableVertexAttribArray(attr_pos);
            glVertexAttribPointer(attr_pos, 3, GL_FLOAT, false, 5 * sizeof(float), (void*)0);
        }

        if (attr_uv != -1)
        {
            glEnableVertexAttribArray(attr_uv);
            glVertexAttribPointer(attr_uv, 2, GL_FLOAT, false, 5 * sizeof(float), (void*)(3 * sizeof(float)));
        }
    }

    d.bindIdx();
    glDrawElements(d.drawMode(), d.getIdxCount(), GL_UNSIGNED_INT, 0);

    RenderingUtils::printGLErrors();
}