"""Self-check for the distance LOD mesher. Run: python test_lod.py

The far levels only earn their polygon saving by claiming they change nothing
you can see. These asserts are that claim, written down:

- Sealing only ever turns air into stone, so no surface can move and no chunk
  can end up thinner than its neighbor expects — a lower level can never open
  a hole in the terrain.
- Every column keeps its topmost block, so the silhouette against the sky is
  identical at every level.
- Air under a tree canopy survives. This is the reason the rule counts solid
  blocks above a cell instead of measuring depth below the column top: a
  canopy is 4 leaf layers, a cave is under the whole crust.

Then it prints the quad/triangle/time table the levels were chosen from.
"""
import time

import numpy as np

from world.blocks import BLOCK_DTYPE, FACE_LAYER, OPAQUE
from world.fast_builder import (AIR, SEAL_COVER, STONE, WATER,
                                build_chunk_mesh_fast, column_seal_limit,
                                make_mesh_buffers, seal_buried_air)
from world.modern_chunk import CHUNK_HEIGHT, CHUNK_SIZE, LEAVES, WOOD
from world.terrain_generator import terrain_generator

PATCH = 5   # chunks per side; the middle ones get real neighbors


def build_patch():
    chunks = {}
    for chunk_x in range(PATCH):
        for chunk_z in range(PATCH):
            blocks = np.zeros((CHUNK_SIZE, CHUNK_HEIGHT, CHUNK_SIZE), dtype=BLOCK_DTYPE)
            terrain_generator.generate_chunk_terrain(chunk_x, chunk_z, blocks)
            chunks[(chunk_x, chunk_z)] = blocks
    return chunks


def neighbors_of(chunks, chunk_x, chunk_z):
    return (chunks[(chunk_x - 1, chunk_z)], chunks[(chunk_x + 1, chunk_z)],
            chunks[(chunk_x, chunk_z - 1)], chunks[(chunk_x, chunk_z + 1)])


def solid(blocks):
    return (blocks != AIR) & (blocks != WATER)


def check_seal_only_adds(blocks):
    """Air may become stone. Nothing else may change."""
    sealed = blocks.copy()
    seal_buried_air(sealed, OPAQUE, SEAL_COVER)

    was_solid = solid(blocks)
    assert np.array_equal(sealed[was_solid], blocks[was_solid]), \
        "sealing changed a block that was already solid"
    changed = sealed != blocks
    assert np.all(sealed[changed] == STONE), "sealing wrote something other than stone"
    assert np.all(solid(sealed) | ~was_solid), "sealing removed solid material"


def check_silhouette_is_kept(blocks):
    """Highest block of every column, before and after."""
    def tops(b):
        s = solid(b)
        return np.where(s.any(axis=1), CHUNK_HEIGHT - 1 - np.argmax(s[:, ::-1, :], axis=1), -1)

    sealed = blocks.copy()
    seal_buried_air(sealed, OPAQUE, SEAL_COVER)
    assert np.array_equal(tops(blocks), tops(sealed)), \
        "a column's highest block moved — the outline against the sky would shift"


def check_trees_survive(chunks):
    """Air with a tree over it must not be sealed: a canopy is thinner than the
    cover the rule asks for."""
    buried_under_tree = 0
    tree_blocks = 0
    for blocks in chunks.values():
        sealed = blocks.copy()
        seal_buried_air(sealed, OPAQUE, SEAL_COVER)
        tree = (blocks == LEAVES) | (blocks == WOOD)
        tree_blocks += int(tree.sum())
        # air that got filled and has a tree block directly above it
        filled = (blocks == AIR) & (sealed == STONE)
        buried_under_tree += int((filled[:, :-1, :] & tree[:, 1:, :]).sum())

    assert tree_blocks > 0, "no trees in the sample — the check proves nothing"
    ratio = buried_under_tree / tree_blocks
    assert ratio < 0.01, f"{ratio:.1%} of tree blocks had the air under them sealed"


def check_column_limit_counts_cover():
    """The limit is 'this many solid blocks above', not 'this far below the top'."""
    blocks = np.zeros((CHUNK_SIZE, CHUNK_HEIGHT, CHUNK_SIZE), dtype=BLOCK_DTYPE)
    blocks[0, 40:40 + SEAL_COVER, 0] = STONE
    assert column_seal_limit(blocks, OPAQUE, 0, 0, SEAL_COVER) == 39, \
        "limit should sit directly under the last of the covering blocks"

    blocks[0, 40 + SEAL_COVER - 1, 0] = AIR          # one short of the cover
    assert column_seal_limit(blocks, OPAQUE, 0, 0, SEAL_COVER) == -1, \
        "a column that never reaches the cover must seal nothing"


def check_footprint(vertices, chunk_x, chunk_z, lod):
    positions = vertices.reshape(-1, 7)[:, :3]
    x_lo, x_hi = chunk_x * CHUNK_SIZE, (chunk_x + 1) * CHUNK_SIZE
    z_lo, z_hi = chunk_z * CHUNK_SIZE, (chunk_z + 1) * CHUNK_SIZE
    assert positions[:, 0].min() == x_lo and positions[:, 0].max() == x_hi, f"lod {lod}: x out of the chunk"
    assert positions[:, 2].min() == z_lo and positions[:, 2].max() == z_hi, f"lod {lod}: z out of the chunk"
    assert positions[:, 1].min() >= 0.0, f"lod {lod}: geometry below y=0"


def main():
    buffers = make_mesh_buffers()
    chunks = build_patch()

    check_column_limit_counts_cover()
    for blocks in chunks.values():
        check_seal_only_adds(blocks)
        check_silhouette_is_kept(blocks)
    check_trees_survive(chunks)

    inner = [(x, z) for x in range(1, PATCH - 1) for z in range(1, PATCH - 1)]
    print(f"{len(inner)} chunks, neighbour-aware, seed {terrain_generator.seed}\n")
    print(f"{'lod':>3} {'quads':>8} {'triangles':>10} {'ms/chunk':>9} {'vs lod0':>8}")

    quads_by_lod = {}
    for lod in (0, 1, 2):
        build_chunk_mesh_fast(chunks[(1, 1)], 1, 1, neighbors_of(chunks, 1, 1),
                              *buffers, FACE_LAYER, OPAQUE, lod)   # warm the JIT

        quads = triangles = 0
        started = time.perf_counter()
        for chunk_x, chunk_z in inner:
            vertices, indices, t_vertices, t_indices = build_chunk_mesh_fast(
                chunks[(chunk_x, chunk_z)], chunk_x, chunk_z,
                neighbors_of(chunks, chunk_x, chunk_z), *buffers,
                FACE_LAYER, OPAQUE, lod)
            assert len(t_vertices) == 0, "generated terrain has no see-through blocks"
            quads += len(vertices) // 28
            triangles += len(indices) // 3
            check_footprint(vertices, chunk_x, chunk_z, lod)
        elapsed = time.perf_counter() - started

        quads_by_lod[lod] = quads
        print(f"{lod:>3} {quads / len(inner):>8.0f} {triangles / len(inner):>10.0f} "
              f"{elapsed / len(inner) * 1000:>9.2f} {quads / quads_by_lod[0]:>7.0%}")

    assert quads_by_lod[1] < quads_by_lod[0], "sealing buried air merged nothing"
    assert quads_by_lod[2] < quads_by_lod[1], "flat shading merged nothing"
    print("\nok")


if __name__ == "__main__":
    main()
