#pragma once

#include <glm/glm.hpp>

struct Vertex
{
    glm::vec3 pos;

    int norUv;
    //   last 3 bits = normal z coord
    //   prev 3 bits = normal y coord
    //        3 bits = normal x coord
    //        9 bits = uv y coord
    //        9 bits = uv x coord
    // -----------------------
    // total 27 bits

    // uv coords = 1/16 block precision
    // normals:
    //   000 = 0
    //   001 = +1
    //   010 = +sqrt(2)/2
    //   101 = -1
    //   110 = -sqrt(2)/2
};
