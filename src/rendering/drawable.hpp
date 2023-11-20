#pragma once

#include <GL/glew.h>

class Drawable
{
protected:
    int idxCount{ -1 };

    GLuint bufIdx;
    GLuint bufVerts;
    GLuint bufFullscreenTriInfo;

public:
    Drawable();
    virtual ~Drawable();

    virtual void bufferVBOs() = 0;
    void destroyVBOs();

    virtual GLenum drawMode() const;
    int getIdxCount() const;

    void generateIdx();
    void generateVerts();
    void generateFullscreenTriInfo();

    bool bindIdx();
    bool bindVerts();
    bool bindFullscreenTriInfo();
};