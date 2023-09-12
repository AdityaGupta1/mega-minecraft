#pragma once

#include "glm/glm.hpp"

using namespace glm;

class Player {
private:
    vec3 pos;

public:
    void tick();

    vec3 getPos();
};