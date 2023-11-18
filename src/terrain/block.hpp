#pragma once

#include <glm/glm.hpp>

enum class Block : unsigned char
{
    AIR,

    WATER,

    BEDROCK,

    STONE,
    DIRT,
    GRASS,
    SAND,
    GRAVEL,
    MYCELIUM,
    SNOW,
    SNOWY_GRASS,
    MUSHROOM_STEM,
    MUSHROOM_UNDERSIDE,
    MUSHROOM_CAP_PURPLE,
    MARBLE,
    ANDESITE,
    CALCITE,
    BLACKSTONE,
    TUFF,
    DEEPSLATE,
    GRANITE,
    SLATE,
    SANDSTONE,
    CLAY,
    RED_SAND,
    RED_SANDSTONE,
    MUD,
    JUNGLE_GRASS,
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
    QUARTZ,
    ICE,
    PACKED_ICE,
    BLUE_ICE,
    SAVANNA_GRASS,
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
    CYPRESS_LEAVES
};

static constexpr int numBlocks = (int)Block::CYPRESS_LEAVES + 1;

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
    OPAQUE,
    SEMI_TRANSPARENT, // e.g. leaves
    TRANSPARENT, // e.g. glass
    X_SHAPED // e.g. flowers, tall grass
};

struct BlockData
{
    BlockUvs uvs;
    TransparencyType transparency{ TransparencyType::OPAQUE };
};

namespace BlockUtils
{
    void init();

    BlockData getBlockData(Block block);
}