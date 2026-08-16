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

from world.blocks import BLOCK_DTYPE, FACE_LAYER, OPAQUE, SHAPE_TABLE
from world.fast_builder import (AIR, SEAL_COVER, STONE, WATER,
                                build_chunk_mesh_fast, column_seal_limit,
                                make_mesh_buffers, seal_buried_air)
from world.modern_chunk import CHUNK_HEIGHT, CHUNK_SIZE
from world.terrain_generator import B_TREES, column_biome, terrain_generator

PATCH = 5   # chunks per side; the middle ones get real neighbors

# Every log and leaf in the block table. The canopy check counts these, and it
# is only meaningful over a patch that has some — which is why the patch below
# is searched for rather than written down: which chunks are forest moves every
# time the terrain is retuned.
TREE_BLOCKS = (7, 38, 39, 40, 41, 42, 43, 127, 133, 134,
               6, 54, 55, 56, 57, 58, 147, 148, 149, 150)


def build_patch():
    origin = (0, 0)
    for chunk_x in range(0, 160, 6):
        for chunk_z in range(0, 160, 6):
            if all(B_TREES[column_biome(x * CHUNK_SIZE + 8, z * CHUNK_SIZE + 8)] > 280
                   for x in range(chunk_x, chunk_x + PATCH)
                   for z in range(chunk_z, chunk_z + PATCH)):
                origin = (chunk_x, chunk_z)
                break
        else:
            continue
        break

    chunks = {}
    for chunk_x in range(PATCH):
        for chunk_z in range(PATCH):
            blocks = np.zeros((CHUNK_SIZE, CHUNK_HEIGHT, CHUNK_SIZE), dtype=BLOCK_DTYPE)
            terrain_generator.generate_chunk_terrain(origin[0] + chunk_x,
                                                     origin[1] + chunk_z, blocks)
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
        tree = np.isin(blocks, TREE_BLOCKS)
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
    print(f"{'lod':>3} {'cube quads':>11} {'plant quads':>12} {'triangles':>10} "
          f"{'ms/chunk':>9} {'vs lod0':>8}")

    quads_by_lod = {}
    shape_quads_by_lod = {}
    for lod in (0, 1, 2):
        build_chunk_mesh_fast(chunks[(1, 1)], 1, 1, neighbors_of(chunks, 1, 1),
                              *buffers, FACE_LAYER, OPAQUE, SHAPE_TABLE, lod)

        quads = shape_quads = triangles = 0
        started = time.perf_counter()
        for chunk_x, chunk_z in inner:
            vertices, indices, t_vertices, t_indices = build_chunk_mesh_fast(
                chunks[(chunk_x, chunk_z)], chunk_x, chunk_z,
                neighbors_of(chunks, chunk_x, chunk_z), *buffers,
                FACE_LAYER, OPAQUE, SHAPE_TABLE, lod)
            quads += len(vertices) // 28
            # The second buffer used to be empty on generated terrain. It is
            # the ground cover now — grass, ferns, flowers — and it is the
            # blended pass, so it is worth watching separately.
            shape_quads += len(t_vertices) // 28
            triangles += (len(indices) + len(t_indices)) // 3
            check_footprint(vertices, chunk_x, chunk_z, lod)
        elapsed = time.perf_counter() - started

        quads_by_lod[lod] = quads
        shape_quads_by_lod[lod] = shape_quads
        print(f"{lod:>3} {quads / len(inner):>11.0f} {shape_quads / len(inner):>12.0f} "
              f"{triangles / len(inner):>10.0f} "
              f"{elapsed / len(inner) * 1000:>9.2f} {quads / quads_by_lod[0]:>7.0%}")

    assert quads_by_lod[1] < quads_by_lod[0], "sealing buried air merged nothing"
    assert quads_by_lod[2] < quads_by_lod[1], "flat shading merged nothing"
    assert shape_quads_by_lod[0] > 0, \
        "no ground cover in the sample — the plant column proves nothing"
    assert shape_quads_by_lod[1] == shape_quads_by_lod[0], \
        "level 1 is supposed to seal caves and nothing else"
    assert shape_quads_by_lod[2] == 0, \
        "level 2 drops ground cover: a tuft of grass is 2.5 px out there"
    print("\nok")


if __name__ == "__main__":
    main()
