#include "renderingUtils.hpp"

#include <GL/glew.h>
#include <iostream>

bool RenderingUtils::printGLErrors()
{
    bool hasError = false;

    GLenum err;
    while ((err = glGetError()) != GL_NO_ERROR)
    {
        hasError = true;
        std::cerr << "OpenGL error: " << err << std::endl;
    }

    return hasError;
}