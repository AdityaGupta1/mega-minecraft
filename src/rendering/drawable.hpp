#pragma once

#include <GL/glew.h>

class Drawable
{
protected:
    int idxCount{ -1 };

    GLuint bufIdx;
    GLuint bufVerts;
    GLuint bufTriInfo;

public:
    Drawable();

    virtual void bufferVBOs() = 0;
    void destroyVBOs();

    virtual GLenum drawMode() const;
    int getIdxCount() const;

    void generateIdx();
    void generateVerts();
    void generateTriInfo();
    void generateUvs();

    bool bindIdx();
    bool bindVerts();
    bool bindTriInfo();
    bool bindUvs();
};