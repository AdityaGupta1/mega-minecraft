#include "utils.hpp"

ivec2 Utils::worldPosToChunkPos(vec3 worldPos)
{
    return ivec2(floor(vec2(worldPos.x, worldPos.z) / 16.f));
}