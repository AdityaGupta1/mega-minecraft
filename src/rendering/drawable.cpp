#include "drawable.hpp"

Drawable::Drawable()
    : bufIdx(), bufVerts()
{
}

void Drawable::destroyVBOs()
{
    glDeleteBuffers(1, &bufIdx);
    glDeleteBuffers(1, &bufVerts);

    idxGenerated = vertsGenerated = false;

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
    idxGenerated = true;
}

void Drawable::generateVerts()
{
    glGenBuffers(1, &bufVerts);
    vertsGenerated = true;
}

bool Drawable::bindIdx()
{
    if (idxGenerated)
    {
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bufIdx);
    }
    return idxGenerated;
}

bool Drawable::bindVerts()
{
    if (vertsGenerated)
    {
        glBindBuffer(GL_ARRAY_BUFFER, bufVerts);
    }
    return vertsGenerated;
}