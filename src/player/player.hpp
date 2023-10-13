#pragma once

#include "glm/glm.hpp"

using namespace glm;

class Player {
private:
    vec3 pos{ 0, 128, -10 };
    float theta{ 0 };
    float phi{ 0 }; // -pi/2 to pi/2

    vec3 forward{ 0, 0, 1 };
    vec3 right{ 1, 0, 0 };
    vec3 up{ 0, 1, 0 };

    mat4 viewMat;

    bool camChanged{ true };

public:
    void tick(bool* viewMatChanged);

    vec3 getPos();
    vec3 getForward();
    vec3 getRight();
    vec3 getUp();

    mat4 getViewMat() const;

    void move(vec3 input);
    void rotate(float dTheta, float dPhi);
};