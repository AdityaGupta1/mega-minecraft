#include "fullscreenTri.hpp"

void FullscreenTri::bufferVBOs()
{
    idxCount = 3;

    GLuint idx[] = { 0, 1, 2 };
    float triInfo[] = {
        -1.f, -1.f, 0.9999999f,        0.f, 0.f,
        3.f, -1.f, 0.9999999f,         2.f, 0.f,
        -1.f, 3.f, 0.9999999f,         0.f, 2.f
    };

    generateIdx();
    bindIdx();
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 3 * sizeof(GLuint), idx, GL_STATIC_DRAW);

    generateFullscreenTriInfo();
    bindFullscreenTriInfo();
    glBufferData(GL_ARRAY_BUFFER, 15 * sizeof(float), triInfo, GL_STATIC_DRAW);
}