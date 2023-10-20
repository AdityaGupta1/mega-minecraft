#pragma once

#include "drawable.hpp"

class FullscreenTri : public Drawable
{
public:
    void bufferVBOs() override;
};