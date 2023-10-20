#include "drawable.hpp"

Drawable::Drawable()
    : bufIdx(-1), bufVerts(-1), bufTriInfo(-1)
{
}

void Drawable::destroyVBOs()
{
    glDeleteBuffers(1, &bufIdx);
    glDeleteBuffers(1, &bufVerts);
    glDeleteBuffers(1, &bufTriInfo);

    bufIdx = bufVerts = bufTriInfo = -1;

    idxCount = -1;
}

GLenum Drawable::drawMode() const
{
    return GL_TRIANGLES;
}

int Drawable::getIdxCount() const
{
    return idxCount;
}

void Drawable::generateIdx()
{
    glGenBuffers(1, &bufIdx);
}

void Drawable::generateVerts()
{
    glGenBuffers(1, &bufVerts);
}

void Drawable::generateTriInfo()
{
    glGenBuffers(1, &bufTriInfo);
}

bool bindBuf(GLenum target, GLuint& buffer)
{
    if (buffer != -1)
    {
        glBindBuffer(target, buffer);
        return true;
    }
    return false;
}

bool Drawable::bindIdx()
{
    return bindBuf(GL_ELEMENT_ARRAY_BUFFER, bufIdx);
}

bool Drawable::bindVerts()
{
    return bindBuf(GL_ARRAY_BUFFER, bufVerts);
}

bool Drawable::bindTriInfo()
{
    return bindBuf(GL_ARRAY_BUFFER, bufTriInfo);
}