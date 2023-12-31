#include "block.hpp"

#include <unordered_map>
#include <array>

using namespace glm;

static std::array<BlockData, numBlocks> blockDatas;

void BlockUtils::init()
{
    blockDatas[(int)Block::AIR] = { BlockUvs(), TransparencyType::T_TRANSPARENT };

    blockDatas[(int)Block::WATER] = { BlockUvs(ivec2(15, 15)), TransparencyType::T_TRANSPARENT };
    blockDatas[(int)Block::LAVA] = { BlockUvs(ivec2(14, 15)), TransparencyType::T_OPAQUE }; // not sure if this should be transparent

    blockDatas[(int)Block::CAVE_VINES_MAIN] = { BlockUvs(ivec2(2, 7)), TransparencyType::T_X_SHAPED };
    blockDatas[(int)Block::CAVE_VINES_GLOW_MAIN] = { BlockUvs(ivec2(3, 7)), TransparencyType::T_X_SHAPED };
    blockDatas[(int)Block::CAVE_VINES_END] = { BlockUvs(ivec2(4, 7)), TransparencyType::T_X_SHAPED };
    blockDatas[(int)Block::CAVE_VINES_GLOW_END] = { BlockUvs(ivec2(5, 7)), TransparencyType::T_X_SHAPED };
    blockDatas[(int)Block::GRASS] = { BlockUvs(ivec2(8, 7)), TransparencyType::T_X_SHAPED };
    blockDatas[(int)Block::JUNGLE_GRASS] = { BlockUvs(ivec2(9, 7)), TransparencyType::T_X_SHAPED };
    blockDatas[(int)Block::SAVANNA_GRASS] = { BlockUvs(ivec2(10, 7)), TransparencyType::T_X_SHAPED };
    blockDatas[(int)Block::WARPED_MUSHROOM] = { BlockUvs(ivec2(9, 5)), TransparencyType::T_X_SHAPED };
    blockDatas[(int)Block::WARPED_ROOTS] = { BlockUvs(ivec2(13, 5)), TransparencyType::T_X_SHAPED };
    blockDatas[(int)Block::NETHER_SPROUTS] = { BlockUvs(ivec2(1, 6)), TransparencyType::T_X_SHAPED };
    blockDatas[(int)Block::INFECTED_MUSHROOM] = { BlockUvs(ivec2(10, 5)), TransparencyType::T_X_SHAPED };
    blockDatas[(int)Block::AMBER_ROOTS] = { BlockUvs(ivec2(4, 6)), TransparencyType::T_X_SHAPED };
    blockDatas[(int)Block::DANDELION] = { BlockUvs(ivec2(11, 7)), TransparencyType::T_X_SHAPED };
    blockDatas[(int)Block::POPPY] = { BlockUvs(ivec2(12, 7)), TransparencyType::T_X_SHAPED };
    blockDatas[(int)Block::PITCHER_BOTTOM] = { BlockUvs(ivec2(13, 7)), TransparencyType::T_X_SHAPED };
    blockDatas[(int)Block::PITCHER_TOP] = { BlockUvs(ivec2(13, 8)), TransparencyType::T_X_SHAPED };
    blockDatas[(int)Block::CORNFLOWER] = { BlockUvs(ivec2(14, 7)), TransparencyType::T_X_SHAPED };
    blockDatas[(int)Block::BLUE_ORCHID] = { BlockUvs(ivec2(15, 7)), TransparencyType::T_X_SHAPED };
    blockDatas[(int)Block::ALLIUM] = { BlockUvs(ivec2(0, 8)), TransparencyType::T_X_SHAPED };
    blockDatas[(int)Block::RED_TULIP] = { BlockUvs(ivec2(1, 8)), TransparencyType::T_X_SHAPED };
    blockDatas[(int)Block::ORANGE_TULIP] = { BlockUvs(ivec2(2, 8)), TransparencyType::T_X_SHAPED };
    blockDatas[(int)Block::WHITE_TULIP] = { BlockUvs(ivec2(3, 8)), TransparencyType::T_X_SHAPED };
    blockDatas[(int)Block::PINK_TULIP] = { BlockUvs(ivec2(4, 8)), TransparencyType::T_X_SHAPED };
    blockDatas[(int)Block::LILAC_BOTTOM] = { BlockUvs(ivec2(5, 8)), TransparencyType::T_X_SHAPED };
    blockDatas[(int)Block::LILAC_TOP] = { BlockUvs(ivec2(5, 9)), TransparencyType::T_X_SHAPED };
    blockDatas[(int)Block::PEONY_BOTTOM] = { BlockUvs(ivec2(6, 8)), TransparencyType::T_X_SHAPED };
    blockDatas[(int)Block::PEONY_TOP] = { BlockUvs(ivec2(6, 9)), TransparencyType::T_X_SHAPED };
    blockDatas[(int)Block::OXEYE_DAISY] = { BlockUvs(ivec2(7, 8)), TransparencyType::T_X_SHAPED };
    blockDatas[(int)Block::LILY_OF_THE_VALLEY] = { BlockUvs(ivec2(8, 8)), TransparencyType::T_X_SHAPED };
    blockDatas[(int)Block::JUNGLE_FERN] = { BlockUvs(ivec2(9, 8)), TransparencyType::T_X_SHAPED };
    blockDatas[(int)Block::SMALL_MAGENTA_CRYSTAL] = { BlockUvs(ivec2(10, 8)), TransparencyType::T_X_SHAPED };
    blockDatas[(int)Block::SMALL_CYAN_CRYSTAL] = { BlockUvs(ivec2(11, 8)), TransparencyType::T_X_SHAPED };
    blockDatas[(int)Block::SMALL_GREEN_CRYSTAL] = { BlockUvs(ivec2(12, 8)), TransparencyType::T_X_SHAPED };
    blockDatas[(int)Block::SMALL_PURPLE_MUSHROOM] = { BlockUvs(ivec2(14, 8)), TransparencyType::T_X_SHAPED };
    blockDatas[(int)Block::DEAD_BUSH] = { BlockUvs(ivec2(15, 8)), TransparencyType::T_X_SHAPED };
    blockDatas[(int)Block::HANGING_SMALL_MAGENTA_CRYSTAL] = { BlockUvs(ivec2(0, 9)), TransparencyType::T_X_SHAPED };
    blockDatas[(int)Block::HANGING_SMALL_CYAN_CRYSTAL] = { BlockUvs(ivec2(1, 9)), TransparencyType::T_X_SHAPED };
    blockDatas[(int)Block::HANGING_SMALL_GREEN_CRYSTAL] = { BlockUvs(ivec2(2, 9)), TransparencyType::T_X_SHAPED };
    blockDatas[(int)Block::TALL_GRASS_BOTTOM] = { BlockUvs(ivec2(3, 9)), TransparencyType::T_X_SHAPED };
    blockDatas[(int)Block::TALL_GRASS_TOP] = { BlockUvs(ivec2(3, 10)), TransparencyType::T_X_SHAPED };
    blockDatas[(int)Block::TALL_JUNGLE_GRASS_BOTTOM] = { BlockUvs(ivec2(4, 9)), TransparencyType::T_X_SHAPED };
    blockDatas[(int)Block::TALL_JUNGLE_GRASS_TOP] = { BlockUvs(ivec2(4, 10)), TransparencyType::T_X_SHAPED };
    blockDatas[(int)Block::TORCHFLOWER] = { BlockUvs(ivec2(7, 9)), TransparencyType::T_X_SHAPED };
    blockDatas[(int)Block::BRAIN_CORAL] = { BlockUvs(ivec2(8, 9)), TransparencyType::T_X_SHAPED };
    blockDatas[(int)Block::BUBBLE_CORAL] = { BlockUvs(ivec2(9, 9)), TransparencyType::T_X_SHAPED };
    blockDatas[(int)Block::FIRE_CORAL] = { BlockUvs(ivec2(10, 9)), TransparencyType::T_X_SHAPED };
    blockDatas[(int)Block::HORN_CORAL] = { BlockUvs(ivec2(11, 9)), TransparencyType::T_X_SHAPED };
    blockDatas[(int)Block::TUBE_CORAL] = { BlockUvs(ivec2(12, 9)), TransparencyType::T_X_SHAPED };
    blockDatas[(int)Block::SEAGRASS] = { BlockUvs(ivec2(13, 9)), TransparencyType::T_X_SHAPED };
    blockDatas[(int)Block::TALL_SEAGRASS_BOTTOM] = { BlockUvs(ivec2(14, 9)), TransparencyType::T_X_SHAPED };
    blockDatas[(int)Block::TALL_SEAGRASS_TOP] = { BlockUvs(ivec2(14, 10)), TransparencyType::T_X_SHAPED };
    blockDatas[(int)Block::KELP_MAIN] = { BlockUvs(ivec2(15, 9)), TransparencyType::T_X_SHAPED };
    blockDatas[(int)Block::KELP_END] = { BlockUvs(ivec2(15, 10)), TransparencyType::T_X_SHAPED };

    blockDatas[(int)Block::BEDROCK] = { BlockUvs(ivec2(0, 5)) };

    blockDatas[(int)Block::STONE] = { BlockUvs(ivec2(3, 0)).setRandomFlip() };
    blockDatas[(int)Block::DIRT] = { BlockUvs(ivec2(0, 0)).setRandomRotation() };
    blockDatas[(int)Block::GRASS_BLOCK] = { BlockUvs(ivec2(1, 0), ivec2(2, 0), ivec2(0, 0)).setRandomRotation(false, true, true) };
    blockDatas[(int)Block::SAND] = { BlockUvs(ivec2(4, 0)).setRandomRotation() };
    blockDatas[(int)Block::GRAVEL] = { BlockUvs(ivec2(5, 0)).setRandomRotation() };
    blockDatas[(int)Block::MYCELIUM] = { BlockUvs(ivec2(6, 0), ivec2(7, 0), ivec2(0, 0)).setRandomRotation(false, true, true) };
    blockDatas[(int)Block::SNOW] = { BlockUvs(ivec2(8, 0)) };
    blockDatas[(int)Block::SNOWY_GRASS_BLOCK] = { BlockUvs(ivec2(9, 0), ivec2(8, 0), ivec2(0, 0)).setRandomRotation(false, true, true) };
    blockDatas[(int)Block::MUSHROOM_STEM] = { BlockUvs(ivec2(10, 0)) };
    blockDatas[(int)Block::MUSHROOM_UNDERSIDE] = { BlockUvs(ivec2(11, 0)).setRandomFlip() };
    blockDatas[(int)Block::PURPLE_MUSHROOM_CAP] = { BlockUvs(ivec2(12, 0)) };
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
    blockDatas[(int)Block::JUNGLE_GRASS_BLOCK] = { BlockUvs(ivec2(0, 2), ivec2(1, 2), ivec2(0, 0)).setRandomRotation(false, true, true) };
    blockDatas[(int)Block::RAFFLESIA_PETAL] = { BlockUvs(ivec2(2, 2)) };
    blockDatas[(int)Block::RAFFLESIA_CENTER] = { BlockUvs(ivec2(3, 2)) };
    blockDatas[(int)Block::RAFFLESIA_SPIKES] = { BlockUvs(ivec2(4, 2)) };
    blockDatas[(int)Block::RAFFLESIA_STEM] = { BlockUvs(ivec2(5, 2)) };
    blockDatas[(int)Block::JUNGLE_WOOD] = { BlockUvs(ivec2(8, 2)) };
    blockDatas[(int)Block::JUNGLE_LEAVES_PLAIN] = { BlockUvs(ivec2(6, 2)), TransparencyType::T_SEMI_TRANSPARENT };
    blockDatas[(int)Block::JUNGLE_LEAVES_FRUITS] = { BlockUvs(ivec2(7, 2)), TransparencyType::T_SEMI_TRANSPARENT };
    blockDatas[(int)Block::CACTUS] = { BlockUvs(ivec2(10, 2)) };
    blockDatas[(int)Block::PALM_WOOD] = { BlockUvs(ivec2(11, 2)) };
    blockDatas[(int)Block::PALM_LEAVES] = { BlockUvs(ivec2(13, 2)), TransparencyType::T_SEMI_TRANSPARENT };
    blockDatas[(int)Block::MAGENTA_CRYSTAL] = { BlockUvs(ivec2(0, 3)), TransparencyType::T_TRANSPARENT };
    blockDatas[(int)Block::CYAN_CRYSTAL] = { BlockUvs(ivec2(1, 3)), TransparencyType::T_TRANSPARENT };
    blockDatas[(int)Block::GREEN_CRYSTAL] = { BlockUvs(ivec2(2, 3)), TransparencyType::T_TRANSPARENT };
    blockDatas[(int)Block::SMOOTH_SAND] = { BlockUvs(ivec2(3, 3)) };
    blockDatas[(int)Block::TERRACOTTA] = { BlockUvs(ivec2(4, 3)) };
    blockDatas[(int)Block::YELLOW_TERRACOTTA] = { BlockUvs(ivec2(5, 3)) };
    blockDatas[(int)Block::ORANGE_TERRACOTTA] = { BlockUvs(ivec2(6, 3)) };
    blockDatas[(int)Block::PURPLE_TERRACOTTA] = { BlockUvs(ivec2(7, 3)) };
    blockDatas[(int)Block::RED_TERRACOTTA] = { BlockUvs(ivec2(8, 3)) };
    blockDatas[(int)Block::WHITE_TERRACOTTA] = { BlockUvs(ivec2(9, 3)) };
    blockDatas[(int)Block::QUARTZ] = { BlockUvs(ivec2(10, 3)) };
    blockDatas[(int)Block::ICE] = { BlockUvs(ivec2(11, 3)), TransparencyType::T_TRANSPARENT };
    blockDatas[(int)Block::PACKED_ICE] = { BlockUvs(ivec2(12, 3)) };
    blockDatas[(int)Block::BLUE_ICE] = { BlockUvs(ivec2(13, 3)) };
    blockDatas[(int)Block::SAVANNA_GRASS_BLOCK] = { BlockUvs(ivec2(14, 2), ivec2(15, 2), ivec2(0, 0)).setRandomRotation(false, true, true) };
    blockDatas[(int)Block::BIRCH_WOOD] = { BlockUvs(ivec2(14, 3)) };
    blockDatas[(int)Block::BIRCH_LEAVES] = { BlockUvs(ivec2(0, 4)), TransparencyType::T_SEMI_TRANSPARENT };
    blockDatas[(int)Block::YELLOW_BIRCH_LEAVES] = { BlockUvs(ivec2(1, 4)), TransparencyType::T_SEMI_TRANSPARENT };
    blockDatas[(int)Block::ORANGE_BIRCH_LEAVES] = { BlockUvs(ivec2(2, 4)), TransparencyType::T_SEMI_TRANSPARENT };
    blockDatas[(int)Block::ACACIA_WOOD] = { BlockUvs(ivec2(3, 4)) };
    blockDatas[(int)Block::ACACIA_LEAVES] = { BlockUvs(ivec2(5, 4)), TransparencyType::T_SEMI_TRANSPARENT };
    blockDatas[(int)Block::SMOOTH_SANDSTONE] = { BlockUvs(ivec2(8, 1)) };
    blockDatas[(int)Block::PINE_WOOD] = { BlockUvs(ivec2(6, 4)) };
    blockDatas[(int)Block::PINE_LEAVES_1] = { BlockUvs(ivec2(8, 4)), TransparencyType::T_SEMI_TRANSPARENT };
    blockDatas[(int)Block::PINE_LEAVES_2] = { BlockUvs(ivec2(9, 4)), TransparencyType::T_SEMI_TRANSPARENT };
    blockDatas[(int)Block::REDWOOD_WOOD] = { BlockUvs(ivec2(10, 4)) };
    blockDatas[(int)Block::REDWOOD_LEAVES] = { BlockUvs(ivec2(12, 4)), TransparencyType::T_SEMI_TRANSPARENT };
    blockDatas[(int)Block::CYPRESS_WOOD] = { BlockUvs(ivec2(13, 4)) };
    blockDatas[(int)Block::CYPRESS_LEAVES] = { BlockUvs(ivec2(15, 4)), TransparencyType::T_SEMI_TRANSPARENT };
    blockDatas[(int)Block::GLOWSTONE] = { BlockUvs(ivec2(1, 5)) };
    blockDatas[(int)Block::SHROOMLIGHT] = { BlockUvs(ivec2(2, 5)) };
    blockDatas[(int)Block::WARPED_DEEPSLATE] = { BlockUvs(ivec2(4, 5), ivec2(3, 5), ivec2(4, 1)).setRandomFlip(false, false, true).setRandomRotation(false, true, false) };
    blockDatas[(int)Block::WARPED_BLACKSTONE] = { BlockUvs(ivec2(5, 5), ivec2(3, 5), ivec2(1, 1)).setRandomRotation(false, true, false) };
    blockDatas[(int)Block::MOSS] = { BlockUvs(ivec2(13, 6)) };
    blockDatas[(int)Block::AMBER_DEEPSLATE] = { BlockUvs(ivec2(7, 5), ivec2(6, 5), ivec2(4, 1)).setRandomFlip(false, false, true).setRandomRotation(false, true, false) };
    blockDatas[(int)Block::AMBER_BLACKSTONE] = { BlockUvs(ivec2(8, 5), ivec2(6, 5), ivec2(1, 1)).setRandomRotation(false, true, false) };
    blockDatas[(int)Block::WARPED_STEM] = { BlockUvs(ivec2(11, 5), ivec2(12, 5)) };
    blockDatas[(int)Block::WARPED_WART] = { BlockUvs(ivec2(0, 6)) };
    blockDatas[(int)Block::AMBER_STEM] = { BlockUvs(ivec2(2, 6), ivec2(3, 6)) };
    blockDatas[(int)Block::AMBER_WART] = { BlockUvs(ivec2(7, 6)) };
    blockDatas[(int)Block::COBBLESTONE] = { BlockUvs(ivec2(6, 7)) };
    blockDatas[(int)Block::COBBLED_DEEPSLATE] = { BlockUvs(ivec2(7, 7)) };
    blockDatas[(int)Block::BRAIN_CORAL_BLOCK] = { BlockUvs(ivec2(8, 10)) };
    blockDatas[(int)Block::BUBBLE_CORAL_BLOCK] = { BlockUvs(ivec2(9, 10)) };
    blockDatas[(int)Block::FIRE_CORAL_BLOCK] = { BlockUvs(ivec2(10, 10)) };
    blockDatas[(int)Block::HORN_CORAL_BLOCK] = { BlockUvs(ivec2(11, 10)) };
    blockDatas[(int)Block::TUBE_CORAL_BLOCK] = { BlockUvs(ivec2(12, 10)) };
    blockDatas[(int)Block::SEA_LANTERN] = { BlockUvs(ivec2(0, 10)) };
}

BlockData BlockUtils::getBlockData(Block block)
{
    return blockDatas[(int)block];
}