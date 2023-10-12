#include "shaderProgram.hpp"

#include "util/utils.hpp"

#include <iostream>

ShaderProgram::ShaderProgram()
    : vertShader(), fragShader(), prog(), attrPos(-1)
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

    attrPos = glGetAttribLocation(prog, "v_Pos");

    std::cout << "done" << std::endl;
    return true;
}

void ShaderProgram::useMe()
{
    glUseProgram(prog);
}