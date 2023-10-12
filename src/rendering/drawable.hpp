#pragma once

#include <GL/glew.h>

class Drawable
{
protected:
    int idxCount{ -1 };

    GLuint bufIdx;
    GLuint bufVerts;

    bool idxGenerated{ false };
    bool vertsGenerated{ false };

public:
    Drawable();

    virtual void createVBOs() = 0;
    void destroyVBOs();

    virtual GLenum drawMode() const;
    int getIdxCount() const;

    void generateIdx();
    void generateVerts();

    bool bindIdx();
    bool bindVerts();
};