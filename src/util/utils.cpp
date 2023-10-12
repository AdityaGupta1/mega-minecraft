#include "utils.hpp"

#include <fstream>
#include <sstream>

ivec2 Utils::worldPosToChunkPos(vec3 worldPos)
{
    return ivec2(floor(vec2(worldPos.x, worldPos.z) / 16.f));
}

std::string Utils::readFile(const std::string& filePath)
{
    std::ifstream stream(filePath.c_str());
    std::stringstream buffer;
    buffer << stream.rdbuf();
    return buffer.str();
}