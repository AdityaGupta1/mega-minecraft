#include "block.hpp"

#include <unordered_map>
#include <array>

using namespace glm;

std::unordered_map<Block, BlockUvs> blockUvs = {
    {Block::STONE, {vec2(3, 0)}},
    {Block::DIRT, {vec2(0, 0)}},
    {Block::GRASS, {vec2(1, 0), vec2(2, 0), vec2(0, 0)}}
};

std::array<BlockData, 256> blockDatas;

void BlockUtils::init()
{
    for (const auto& entry : blockUvs)
    {
        BlockUvs uvs = entry.second;
        uvs.normalize();
        blockDatas[(int)entry.first].uvs = uvs;
    }

    blockUvs.clear();
}

BlockData BlockUtils::getBlockData(Block block)
{
    return blockDatas[(int)block];
}