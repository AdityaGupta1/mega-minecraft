#include "block.hpp"

#include <unordered_map>
#include <array>

using namespace glm;

std::unordered_map<Block, BlockUvs> blockUvs = {
    {Block::STONE, BlockUvs(ivec2(3, 0)).setRandomFlip()},
    {Block::DIRT, BlockUvs(ivec2(0, 0)).setRandomRotation()},
    {Block::GRASS, BlockUvs(ivec2(1, 0), ivec2(2, 0), ivec2(0, 0)).setRandomRotation(false, true, true)},
    {Block::SAND, BlockUvs(ivec2(4, 0)).setRandomRotation()},
    {Block::GRAVEL, BlockUvs(ivec2(5, 0)).setRandomRotation()},
    {Block::MYCELIUM, BlockUvs(ivec2(6, 0), ivec2(7, 0), ivec2(0, 0)).setRandomRotation(false, true, true)},
    {Block::SNOW, BlockUvs(ivec2(8, 0)).setRandomRotation()},
    {Block::SNOWY_GRASS, BlockUvs(ivec2(9, 0), ivec2(8, 0), ivec2(0, 0)).setRandomRotation(false, true, true)},
    {Block::MUSHROOM_STEM, BlockUvs(ivec2(10, 0))},
    {Block::MUSHROOM_UNDERSIDE, BlockUvs(ivec2(11, 0)).setRandomRotation()},
    {Block::MUSHROOM_CAP_PURPLE, BlockUvs(ivec2(12, 0))}
};

std::array<BlockData, 256> blockDatas;

void BlockUtils::init()
{
    for (const auto& entry : blockUvs)
    {
        blockDatas[(int)entry.first].uvs = entry.second;
    }

    blockUvs.clear();
}

BlockData BlockUtils::getBlockData(Block block)
{
    return blockDatas[(int)block];
}