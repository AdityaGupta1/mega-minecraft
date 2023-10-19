#include "block.hpp"

#include <unordered_map>
#include <array>

using namespace glm;

std::unordered_map<Block, BlockUvs> blockUvs = {
    {Block::STONE, BlockUvs(vec2(3, 0)).setRandomFlip()},
    {Block::DIRT, BlockUvs(vec2(0, 0)).setRandomRotation()},
    {Block::GRASS, BlockUvs(vec2(1, 0), vec2(2, 0), vec2(0, 0)).setRandomRotation(false, true, true)},
    {Block::SAND, BlockUvs(vec2(4, 0)).setRandomRotation()},
    {Block::GRAVEL, BlockUvs(vec2(5, 0)).setRandomRotation()},
    {Block::MYCELIUM, BlockUvs(vec2(6, 0), vec2(7, 0), vec2(0, 0)).setRandomRotation(false, true, true)},
    {Block::SNOW, BlockUvs(vec2(8, 0)).setRandomRotation()},
    {Block::SNOWY_GRASS, BlockUvs(vec2(9, 0), vec2(8, 0), vec2(0, 0)).setRandomRotation(false, true, true)},
    {Block::MUSHROOM_STEM, BlockUvs(vec2(10, 0))},
    {Block::MUSHROOM_UNDERSIDE, BlockUvs(vec2(11, 0)).setRandomRotation()},
    {Block::MUSHROOM_CAP_PURPLE, BlockUvs(vec2(12, 0))}
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