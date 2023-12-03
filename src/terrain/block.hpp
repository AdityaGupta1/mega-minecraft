#pragma once

#include <glm/glm.hpp>

enum class Block : unsigned char
{
    AIR,

    WATER,
    LAVA,

    CAVE_VINES_MAIN,
    CAVE_VINES_GLOW_MAIN,
    CAVE_VINES_END,
    CAVE_VINES_GLOW_END,
    GRASS,
    JUNGLE_GRASS,
    SAVANNA_GRASS,
    WARPED_MUSHROOM,
    WARPED_ROOTS,
    NETHER_SPROUTS,
    INFECTED_MUSHROOM,
    AMBER_ROOTS,
    DANDELION,
    POPPY,
    PITCHER_BOTTOM, 
    PITCHER_TOP,
    CORNFLOWER,
    BLUE_ORCHID,
    ALLIUM,
    RED_TULIP,
    ORANGE_TULIP,
    WHITE_TULIP,
    PINK_TULIP,
    LILAC_BOTTOM,
    LILAC_TOP,
    PEONY_BOTTOM,
    PEONY_TOP,
    OXEYE_DAISY,
    LILY_OF_THE_VALLEY,
    JUNGLE_FERN,
    SMALL_MAGENTA_CRYSTAL,
    SMALL_CYAN_CRYSTAL,
    SMALL_GREEN_CRYSTAL,
    SMALL_PURPLE_MUSHROOM,
    DEAD_BUSH,
    HANGING_SMALL_MAGENTA_CRYSTAL,
    HANGING_SMALL_CYAN_CRYSTAL,
    HANGING_SMALL_GREEN_CRYSTAL,
    TALL_GRASS_BOTTOM,
    TALL_GRASS_TOP,
    TALL_JUNGLE_GRASS_BOTTOM,
    TALL_JUNGLE_GRASS_TOP,
    TORCHFLOWER,

    BEDROCK,

    STONE,
    DIRT,
    GRASS_BLOCK,
    SAND,                   // rough microfacet
    GRAVEL,                 // rough microfacet
    MYCELIUM,
    SNOW,
    SNOWY_GRASS_BLOCK,
    MUSHROOM_STEM,
    MUSHROOM_UNDERSIDE,
    PURPLE_MUSHROOM_CAP,
    MARBLE,                 // smooth microfacet
    ANDESITE,               // smooth microfacet
    CALCITE,                // smooth microfacet
    BLACKSTONE,
    TUFF,
    DEEPSLATE,              // smooth microfacet
    GRANITE,                // smooth microfacet
    SLATE,                  // smooth microfacet
    SANDSTONE,
    CLAY,
    RED_SAND,
    RED_SANDSTONE,
    MUD,
    JUNGLE_GRASS_BLOCK,
    RAFFLESIA_PETAL,
    RAFFLESIA_CENTER,
    RAFFLESIA_SPIKES,
    RAFFLESIA_STEM,
    JUNGLE_WOOD,
    JUNGLE_LEAVES_PLAIN,
    JUNGLE_LEAVES_FRUITS,
    CACTUS,
    PALM_WOOD,
    PALM_LEAVES,
    MAGENTA_CRYSTAL,
    CYAN_CRYSTAL,
    GREEN_CRYSTAL,
    SMOOTH_SAND,
    TERRACOTTA,
    YELLOW_TERRACOTTA,
    ORANGE_TERRACOTTA,
    PURPLE_TERRACOTTA,
    RED_TERRACOTTA,
    WHITE_TERRACOTTA,
    QUARTZ,                 // smooth microfacet
    ICE,                    // smooth microfacet
    PACKED_ICE,             // smooth microfacet
    BLUE_ICE,               // smooth microfacet
    SAVANNA_GRASS_BLOCK,
    BIRCH_WOOD,
    BIRCH_LEAVES,
    YELLOW_BIRCH_LEAVES,
    ORANGE_BIRCH_LEAVES,
    ACACIA_WOOD,
    ACACIA_LEAVES,
    SMOOTH_SANDSTONE,
    PINE_WOOD,
    PINE_LEAVES_1,
    PINE_LEAVES_2,
    REDWOOD_WOOD,
    REDWOOD_LEAVES,
    CYPRESS_WOOD,
    CYPRESS_LEAVES,
    GLOWSTONE,
    SHROOMLIGHT,
    WARPED_DEEPSLATE,
    WARPED_BLACKSTONE,
    MOSS,
    AMBER_DEEPSLATE,
    AMBER_BLACKSTONE,
    WARPED_STEM,
    WARPED_WART,
    AMBER_STEM,
    AMBER_WART,
    COBBLESTONE,
    COBBLED_DEEPSLATE
};

static constexpr int numBlocks = (int)Block::COBBLED_DEEPSLATE + 1;
static constexpr int numNonSolidBlocks = (int)Block::TORCHFLOWER + 1;

struct SideUv
{
    SideUv() = default;
    SideUv(glm::ivec2 uv) : uv(uv) {}

    glm::ivec2 uv{ 0 };
    bool randRot{ false };
    bool randFlip{ false };
};

struct BlockUvs
{
    BlockUvs() = default;
    BlockUvs(glm::vec2 all) : top(all), side(all), bottom(all) {}
    BlockUvs(glm::vec2 side, glm::vec2 vert) : top(vert), side(side), bottom(vert) {}
    BlockUvs(glm::vec2 side, glm::vec2 top, glm::vec2 bottom) : top(top), side(side), bottom(bottom) {}

    SideUv side;
    SideUv top;
    SideUv bottom;

    BlockUvs& setRandomRotation()
    {
        side.randRot = top.randRot = bottom.randRot = true;
        return *this;
    }

    BlockUvs& setRandomRotation(bool side, bool top, bool bottom)
    {
        this->side.randRot = side;
        this->top.randRot = top;
        this->bottom.randRot = bottom;
        return *this;
    }

    BlockUvs& setRandomFlip()
    {
        side.randFlip = top.randFlip = bottom.randFlip = true;
        return *this;
    }

    BlockUvs& setRandomFlip(bool side, bool top, bool bottom)
    {
        this->side.randFlip = side;
        this->top.randFlip = top;
        this->bottom.randFlip = bottom;
        return *this;
    }
};

enum class TransparencyType : unsigned char
{
    T_OPAQUE,
    T_SEMI_TRANSPARENT, // e.g. leaves
    T_TRANSPARENT, // e.g. glass
    T_X_SHAPED // e.g. flowers, tall grass
};

struct BlockData
{
    BlockUvs uvs;
    TransparencyType transparency{ TransparencyType::T_OPAQUE };
};

namespace BlockUtils
{
    void init();

    BlockData getBlockData(Block block);
}