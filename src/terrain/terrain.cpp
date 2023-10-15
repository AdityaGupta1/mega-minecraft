#include "terrain.hpp"

#include "util/enums.hpp"

#define CHUNK_VBOS_GEN_RADIUS 3
#define CHUNK_FEATURE_PLACEMENTS_GEN_RADIUS (CHUNK_VBOS_GEN_RADIUS + 2)

Terrain::Terrain()
{}

ivec2 zonePosFromChunkPos(ivec2 chunkPos)
{
    return ivec2(glm::floor(vec2(chunkPos) / 16.f)) * 16;
}

Zone* Terrain::createZone(ivec2 zonePos)
{
    auto newZoneUptr = std::make_unique<Zone>(zonePos);
    Zone* newZonePtr = newZoneUptr.get();
    zones[zonePos] = std::move(newZoneUptr);

    for (int i = 0; i < 8; ++i)
    {
        ivec2 neighborPos = zonePos + (16 * DirectionEnums::dirVecs2D[i]);

        auto neighborZoneIt = zones.find(neighborPos);
        if (neighborZoneIt == zones.end())
        {
            continue;
        }

        auto neighborZonePtr = neighborZoneIt->second.get();

        newZonePtr->neighbors[i] = neighborZonePtr;
        neighborZonePtr->neighbors[(i + 4) % 8] = newZonePtr;
    }

    return newZonePtr;
}

// This function looks at the area covered by CHUNK_FEATURE_PLACEMENTS_GEN_RADIUS. For each chunk, it first
// creates the chunk if it doesn't exist, also creating the associated zone if necsesary. Then, it looks at
// the chunk's state and adds it to the correct queue iff chunk.readyForQueue == true. This also includes
// setting readyForQueue to false so the chunk will be skipped until it's ready for the next state. The states 
// themselves, as well as readyForQueue, will be updated after the associated call or CPU thread finishes 
// execution. Kernels and threads are spawned from Terrain::tick().
void Terrain::updateChunks()
{
    for (int dz = -CHUNK_FEATURE_PLACEMENTS_GEN_RADIUS; dz <= CHUNK_FEATURE_PLACEMENTS_GEN_RADIUS; ++dz)
    {
        for (int dx = -CHUNK_FEATURE_PLACEMENTS_GEN_RADIUS; dx <= CHUNK_FEATURE_PLACEMENTS_GEN_RADIUS; ++dx)
        {
            ivec2 newChunkWorldPos = currentChunkPos + ivec2(dx, dz);
            ivec2 newZoneWorldPos = zonePosFromChunkPos(newChunkWorldPos);

            auto zoneIt = zones.find(newZoneWorldPos);
            Zone* zonePtr; // TODO maybe cache this and reuse if next chunk has same zone (which should be a common occurrence)
            if (zoneIt == zones.end())
            {
                zonePtr = createZone(newZoneWorldPos);
            }
            else
            {
                zonePtr = zoneIt->second.get();
            }

            ivec2 newChunkLocalPos = newChunkWorldPos - newZoneWorldPos;
            int chunkIdx = newChunkLocalPos.x + 16 * newChunkLocalPos.y;

            if (zonePtr->chunks[chunkIdx] == nullptr)
            {
                auto chunkUptr = std::make_unique<Chunk>(newChunkWorldPos);
                chunkUptr->zonePtr = zonePtr;
                zonePtr->chunks[chunkIdx] = std::move(chunkUptr);
            }

            Chunk* chunkPtr = zonePtr->chunks[chunkIdx].get();

            // TODO add chunk to appropriate queue (based on current state) if ready
            // don't go past feature placements if not in range of CHUNK_VBOS_GEN_RADIUS
        }
    }
}

void Terrain::tick()
{
    updateChunks();

    // TODO do a kernel or launch a thread or something (based on queue of chunks to generate)
    // go through all the queues and do kernels/threads (up to a max number of kernels, probably no limit on queueing threads)
    // max number of kernels may depend on type and measured execution type of each kernel
    // make sure to update chunk.readyForQueue to true afterwards
    // IMPORTANT: VBO threads should, when finished, directly add chunks to drawableChunks to prevent leaking chunks that become too far
}

void Terrain::draw() {
    // TODO draw things lol

    // when iterating through collection of chunks to draw, keep track of which ones are too far, queue them for destruction, and skip them
    // maybe the chunks can be destroyed in tick() before calling updateChunks()
}

void Terrain::setCurrentChunkPos(ivec2 newCurrentChunk) 
{
    this->currentChunkPos = newCurrentChunk;
}