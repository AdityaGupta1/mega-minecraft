#pragma once

#include <glm/glm.hpp>

struct Vertex
{
    glm::vec3 pos;
    glm::vec3 nor; // TODO: compact this
    glm::vec2 uv;
};
