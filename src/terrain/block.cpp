#include "block.hpp"

#include <unordered_map>
#include <array>

using namespace glm;

std::array<BlockData, numBlocks> blockDatas;

void BlockUtils::init()
{
    blockDatas[(int)Block::STONE] = { BlockUvs(ivec2(3, 0)).setRandomFlip() };
    blockDatas[(int)Block::DIRT] = { BlockUvs(ivec2(0, 0)).setRandomRotation() };
    blockDatas[(int)Block::GRASS] = { BlockUvs(ivec2(1, 0), ivec2(2, 0), ivec2(0, 0)).setRandomRotation(false, true, true) };
    blockDatas[(int)Block::SAND] = { BlockUvs(ivec2(4, 0)).setRandomRotation() };
    blockDatas[(int)Block::GRAVEL] = { BlockUvs(ivec2(5, 0)).setRandomRotation() };
    blockDatas[(int)Block::MYCELIUM] = { BlockUvs(ivec2(6, 0), ivec2(7, 0), ivec2(0, 0)).setRandomRotation(false, true, true) };
    blockDatas[(int)Block::SNOW] = { BlockUvs(ivec2(8, 0)).setRandomRotation() };
    blockDatas[(int)Block::SNOWY_GRASS] = { BlockUvs(ivec2(9, 0), ivec2(8, 0), ivec2(0, 0)).setRandomRotation(false, true, true) };
    blockDatas[(int)Block::MUSHROOM_STEM] = { BlockUvs(ivec2(10, 0)) };
    blockDatas[(int)Block::MUSHROOM_UNDERSIDE] = { BlockUvs(ivec2(11, 0)).setRandomFlip() };
    blockDatas[(int)Block::MUSHROOM_CAP_PURPLE] = { BlockUvs(ivec2(12, 0)) };
    blockDatas[(int)Block::MARBLE] = { BlockUvs(ivec2(13, 0)).setRandomRotation() };
    blockDatas[(int)Block::ANDESITE] = { BlockUvs(ivec2(14, 0)) };
    blockDatas[(int)Block::CALCITE] = { BlockUvs(ivec2(15, 0)) };
    blockDatas[(int)Block::BLACKSTONE] = { BlockUvs(ivec2(0, 1), ivec2(1, 1)) };
    blockDatas[(int)Block::TUFF] = { BlockUvs(ivec2(2, 1)) };
    blockDatas[(int)Block::DEEPSLATE] = { BlockUvs(ivec2(3, 1), ivec2(4, 1)).setRandomFlip(false, true, true) };
    blockDatas[(int)Block::GRANITE] = { BlockUvs(ivec2(5, 1)).setRandomRotation() };
    blockDatas[(int)Block::SLATE] = { BlockUvs(ivec2(6, 1)) };
    blockDatas[(int)Block::SANDSTONE] = { BlockUvs(ivec2(7, 1), ivec2(8, 1), ivec2(9, 1)) };
    blockDatas[(int)Block::CLAY] = { BlockUvs(ivec2(10, 1)) };
    blockDatas[(int)Block::RED_SAND] = { BlockUvs(ivec2(11, 1)).setRandomRotation() };
    blockDatas[(int)Block::RED_SANDSTONE] = { BlockUvs(ivec2(12, 1), ivec2(13, 1), ivec2(14, 1)) };
    blockDatas[(int)Block::MUD] = { BlockUvs(ivec2(15, 1)) };
    blockDatas[(int)Block::JUNGLE_GRASS] = { BlockUvs(ivec2(0, 2), ivec2(1, 2), ivec2(0, 0)).setRandomRotation(false, true, true) };
    blockDatas[(int)Block::RAFFLESIA_PETAL] = { BlockUvs(ivec2(2, 2)) };
}

BlockData BlockUtils::getBlockData(Block block)
{
    return blockDatas[(int)block];
}