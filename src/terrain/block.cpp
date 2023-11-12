#include "block.hpp"

#include <unordered_map>
#include <array>

using namespace glm;

static std::array<BlockData, numBlocks> blockDatas;

void BlockUtils::init()
{
    blockDatas[(int)Block::AIR] = { BlockUvs(), TransparencyType::TRANSPARENT };

    blockDatas[(int)Block::WATER] = { BlockUvs(ivec2(15, 15)), TransparencyType::TRANSPARENT };

    blockDatas[(int)Block::STONE] = { BlockUvs(ivec2(3, 0)).setRandomFlip() };
    blockDatas[(int)Block::DIRT] = { BlockUvs(ivec2(0, 0)).setRandomRotation() };
    blockDatas[(int)Block::GRASS] = { BlockUvs(ivec2(1, 0), ivec2(2, 0), ivec2(0, 0)).setRandomRotation(false, true, true) };
    blockDatas[(int)Block::SAND] = { BlockUvs(ivec2(4, 0)).setRandomRotation() };
    blockDatas[(int)Block::GRAVEL] = { BlockUvs(ivec2(5, 0)).setRandomRotation() };
    blockDatas[(int)Block::MYCELIUM] = { BlockUvs(ivec2(6, 0), ivec2(7, 0), ivec2(0, 0)).setRandomRotation(false, true, true) };
    blockDatas[(int)Block::SNOW] = { BlockUvs(ivec2(8, 0)) };
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
    blockDatas[(int)Block::RAFFLESIA_CENTER] = { BlockUvs(ivec2(3, 2)) };
    blockDatas[(int)Block::RAFFLESIA_SPIKES] = { BlockUvs(ivec2(4, 2)) };
    blockDatas[(int)Block::RAFFLESIA_STEM] = { BlockUvs(ivec2(5, 2)) };
    blockDatas[(int)Block::JUNGLE_WOOD] = { BlockUvs(ivec2(8, 2)) };
    blockDatas[(int)Block::JUNGLE_LEAVES_PLAIN] = { BlockUvs(ivec2(6, 2)), TransparencyType::SEMI_TRANSPARENT };
    blockDatas[(int)Block::JUNGLE_LEAVES_FRUITS] = { BlockUvs(ivec2(7, 2)), TransparencyType::SEMI_TRANSPARENT };
    blockDatas[(int)Block::CACTUS] = { BlockUvs(ivec2(10, 2)) };
    blockDatas[(int)Block::PALM_WOOD] = { BlockUvs(ivec2(11, 2)) };
    blockDatas[(int)Block::PALM_LEAVES] = { BlockUvs(ivec2(13, 2)), TransparencyType::SEMI_TRANSPARENT };
    blockDatas[(int)Block::MAGENTA_CRYSTAL] = { BlockUvs(ivec2(0, 3)), TransparencyType::TRANSPARENT };
    blockDatas[(int)Block::CYAN_CRYSTAL] = { BlockUvs(ivec2(1, 3)), TransparencyType::TRANSPARENT };
    blockDatas[(int)Block::GREEN_CRYSTAL] = { BlockUvs(ivec2(2, 3)), TransparencyType::TRANSPARENT };
    blockDatas[(int)Block::SMOOTH_SAND] = { BlockUvs(ivec2(3, 3)) };
    blockDatas[(int)Block::TERRACOTTA] = { BlockUvs(ivec2(4, 3)) };
    blockDatas[(int)Block::YELLOW_TERRACOTTA] = { BlockUvs(ivec2(5, 3)) };
    blockDatas[(int)Block::ORANGE_TERRACOTTA] = { BlockUvs(ivec2(6, 3)) };
    blockDatas[(int)Block::PURPLE_TERRACOTTA] = { BlockUvs(ivec2(7, 3)) };
    blockDatas[(int)Block::RED_TERRACOTTA] = { BlockUvs(ivec2(8, 3)) };
    blockDatas[(int)Block::WHITE_TERRACOTTA] = { BlockUvs(ivec2(9, 3)) };
    blockDatas[(int)Block::QUARTZ] = { BlockUvs(ivec2(10, 3)) };
    blockDatas[(int)Block::ICE] = { BlockUvs(ivec2(11, 3)), TransparencyType::TRANSPARENT };
    blockDatas[(int)Block::PACKED_ICE] = { BlockUvs(ivec2(12, 3)) };
    blockDatas[(int)Block::BLUE_ICE] = { BlockUvs(ivec2(13, 3)) };
    blockDatas[(int)Block::SAVANNA_GRASS] = { BlockUvs(ivec2(14, 2), ivec2(15, 2), ivec2(0, 0)).setRandomRotation(false, true, true) };
    blockDatas[(int)Block::BIRCH_WOOD] = { BlockUvs(ivec2(14, 3)) };
    blockDatas[(int)Block::BIRCH_LEAVES] = { BlockUvs(ivec2(0, 4)), TransparencyType::SEMI_TRANSPARENT };
    blockDatas[(int)Block::YELLOW_BIRCH_LEAVES] = { BlockUvs(ivec2(1, 4)), TransparencyType::SEMI_TRANSPARENT };
}

BlockData BlockUtils::getBlockData(Block block)
{
    return blockDatas[(int)block];
}