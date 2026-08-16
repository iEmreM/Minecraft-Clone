"""Self-check for the see-through blocks. Run: python test_transparency.py

Transparency fails as *looks wrong*, never as an exception, and it fails in
three different places:

  * the mesher has to route a block's quads to the pass that can draw it, drop
    the faces a glass block shares with its own kind, and keep the faces of the
    terrain behind it — the whole point of the feature;
  * `texture.png` has to still carry alpha on those layers. `build_atlas.py`
    flattens every other block against its own average, and a rebake that took
    that path for glass would produce a solid pale block with no error;
  * the two chunk programs have to keep the same uniforms. They share a vertex
    shader and a body, and a uniform written to one and not the other fogs a
    window differently from the wall it is set into.

Nothing here needs a window: the mesher is plain numpy, and the shaders compile
in a standalone context.
"""

import re

import moderngl
import numpy as np
import pygame as pg

from engine.shader_manager import ShaderManager, load_source
from world import blocks
from world.blocks import BLOCK_DTYPE, FACE_LAYER, OPAQUE
from world.fast_builder import (CHUNK_HEIGHT, CHUNK_SIZE, MAX_FACES_ALPHA,
                                NO_NEIGHBOR, build_chunk_mesh_fast,
                                make_mesh_buffers)

GLASS = min(blocks.TRANSPARENT)
OTHER_GLASS = sorted(blocks.TRANSPARENT)[1]
STONE = 3
DIRT = 2
ATLAS = 'texture.png'
TILE = 16

BUFFERS = make_mesh_buffers()
EMPTY = (NO_NEIGHBOR,) * 4


def code(path):
    """A shader with its includes resolved and its comments stripped.

    Comments matter here: both files talk about `discard` at length, and the
    check below is about what the driver compiles, not what the file says.
    """
    source = load_source(path)
    source = re.sub(r'/\*.*?\*/', '', source, flags=re.S)
    return re.sub(r'//[^\n]*', '', source)


def mesh(grid):
    """(opaque quads, see-through quads) for a chunk of blocks."""
    v, _, tv, _ = build_chunk_mesh_fast(grid, 0, 0, EMPTY, *BUFFERS,
                                        FACE_LAYER, OPAQUE, 0)
    return len(v) // 28, len(tv) // 28


def world():
    return np.zeros((CHUNK_SIZE, CHUNK_HEIGHT, CHUNK_SIZE), dtype=BLOCK_DTYPE)


def check_registry():
    assert blocks.TRANSPARENT, 'no transparent blocks'
    assert OPAQUE[0] == 0, 'AIR must not hide a face'
    assert OPAQUE[8] == 0, 'WATER must not hide a face'
    for bid in blocks.BLOCK_NAMES:
        want = 0 if bid in blocks.TRANSPARENT else 1
        assert bool(OPAQUE[bid]) == bool(want), \
            f'{blocks.BLOCK_NAMES[bid]} is on the wrong side'
        # OPAQUE is a three-value enum, not a flag: foliage hides the face
        # behind it like any solid block but must not count as cover for the far
        # LOD's cave sealing, so it carries a 2. Only column_seal_limit reads it.
        assert (OPAQUE[bid] == 2) == (bid in blocks.FOLIAGE and bid not in
                                      blocks.TRANSPARENT), \
            f'{blocks.BLOCK_NAMES[bid]} disagrees with the Yaprak group'
    print(f'registry: {len(blocks.TRANSPARENT)} see-through of '
          f'{len(blocks.BLOCK_NAMES)} blocks')


def check_atlas_keeps_alpha():
    """The see-through layers still have alpha; every other layer has none."""
    surface = pg.image.load(ATLAS)
    assert surface.get_flags() & pg.SRCALPHA or surface.get_bytesize() == 4, \
        f'{ATLAS} has no alpha channel — rerun `python build_atlas.py`'
    alpha = pg.surfarray.array_alpha(surface)

    see_through = {name for bid in blocks.TRANSPARENT for name in blocks.BLOCK_FACES[bid]}
    for layer, name in enumerate(blocks.TEXTURES):
        tile = alpha[:, layer * TILE:(layer + 1) * TILE]
        if name in see_through:
            assert tile.min() < 250, f'{name}: baked opaque, the block would be solid'
        else:
            assert tile.min() == 255, f'{name}: kept alpha, the opaque pass would blend it'
    print(f'{ATLAS}: {len(see_through)} layers keep their alpha, '
          f'{len(blocks.TEXTURES) - len(see_through)} are flat')


def check_glass_meshes_into_the_second_buffer():
    grid = world()
    grid[4, 10, 4] = GLASS
    opaque, alpha = mesh(grid)
    assert opaque == 0, 'a glass block put geometry in the opaque pass'
    assert alpha == 6, f'a lone glass block should be 6 quads, got {alpha}'

    grid[4, 10, 4] = STONE
    opaque, alpha = mesh(grid)
    assert (opaque, alpha) == (6, 0), 'a stone block leaked into the see-through pass'


def check_glass_hides_its_own_kind():
    """Two glass blocks touching share no face; two different ones do.

    Counting quads rather than faces, so the numbers are what the GPU is asked
    to draw: a second block of the same kind adds nothing at all — the seam is
    dropped and the four side runs simply grow one block longer.
    """
    grid = world()
    grid[4, 10, 4] = grid[5, 10, 4] = GLASS
    _, alpha = mesh(grid)
    assert alpha == 6, f'the seam between two glass blocks was drawn ({alpha} quads)'

    grid[5, 10, 4] = OTHER_GLASS
    _, alpha = mesh(grid)
    assert alpha == 12, 'glass against a different colour must keep both faces'


def check_terrain_behind_glass_survives():
    """A solid block covered by glass keeps the face under it — that face is
    what you actually see through the window. DIRT is the control: two solid
    blocks do hide the face between them, and being a different type from STONE
    it cannot merge with it, so the quad count stays readable."""
    grid = world()
    grid[4, 10, 4] = STONE
    alone, _ = mesh(grid)
    assert alone == 6

    grid[4, 11, 4] = GLASS
    with_glass, alpha = mesh(grid)
    assert with_glass == alone, 'glass hid the face of the block under it'
    assert alpha == 5, 'the glass face touching the stone should be dropped'

    grid[4, 11, 4] = DIRT
    with_dirt, alpha = mesh(grid)
    assert alpha == 0, 'dirt is not see-through'
    assert with_dirt == 2 * alone - 2, \
        'two solid blocks are supposed to hide the face between them'


def check_glass_casts_no_ambient_occlusion():
    """A pane beside a floor must not draw a contact shadow under itself."""
    floor = world()
    floor[:, 9, :] = STONE
    bare = build_chunk_mesh_fast(floor, 0, 0, EMPTY, *BUFFERS, FACE_LAYER, OPAQUE, 0)[0]

    floor[4, 10, 4] = GLASS
    glazed = build_chunk_mesh_fast(floor, 0, 0, EMPTY, *BUFFERS, FACE_LAYER, OPAQUE, 0)[0]
    assert np.array_equal(bare, glazed), 'the glass darkened the floor around it'

    floor[4, 10, 4] = STONE
    shaded = build_chunk_mesh_fast(floor, 0, 0, EMPTY, *BUFFERS, FACE_LAYER, OPAQUE, 0)[0]
    assert not np.array_equal(bare, shaded), \
        'a solid block is supposed to occlude — this check proves nothing otherwise'


def check_overflow_drops_glass_not_terrain():
    """The see-through buffer is the small one, so it is the one that gives up.

    A checkerboard of alternating glass colours merges into nothing and blows
    past MAX_FACES_ALPHA in a single chunk. Stopping the whole sweep there —
    which is what the opaque buffer's guard does — would trade a wall of glass
    for a hole in the world.
    """
    grid = world()
    grid[:, :20, :] = STONE
    ids = sorted(blocks.TRANSPARENT)
    for x in range(CHUNK_SIZE):
        for z in range(CHUNK_SIZE):
            for y in range(20, 120):
                if (x + y + z) % 2 == 0:
                    grid[x, y, z] = ids[(x + y + z) % len(ids)]

    opaque, alpha = mesh(grid)
    assert alpha == MAX_FACES_ALPHA, f'the cap did not engage ({alpha} quads)'
    assert opaque > 0, 'the terrain went missing when the glass buffer filled'
    print(f'overflow: {alpha} see-through quads capped, {opaque} opaque quads kept')


def check_programs_agree():
    """Both chunk programs take the same uniforms, and only the alpha one may
    contain a discard."""
    ctx = moderngl.create_standalone_context()
    manager = ShaderManager(ctx)
    manager.load_default_shaders()
    opaque = manager.get_program('chunk')
    alpha = manager.get_program('chunk_alpha')
    assert opaque is not None and alpha is not None, 'a chunk shader failed to compile'

    shared = {'m_proj', 'm_view', 'cam_pos', 'fog_range', 'water_line',
              'u_texture_0', 'sky_horizon', 'sky_zenith', 'u_time'}
    for name in shared:
        assert name in opaque and name in alpha, f'{name} is missing from one program'

    assert 'discard' not in code('shaders/chunk.frag'), \
        'the opaque terrain pass must not discard — it costs early-Z for the whole world'
    assert 'discard' in code('shaders/chunk_alpha.frag'), \
        'without a discard the holes in glass would still write depth'

    ctx.release()
    print(f'both chunk programs carry all {len(shared)} shared uniforms')


def main():
    check_registry()
    check_atlas_keeps_alpha()
    check_glass_meshes_into_the_second_buffer()
    check_glass_hides_its_own_kind()
    check_terrain_behind_glass_survives()
    check_glass_casts_no_ambient_occlusion()
    check_overflow_drops_glass_not_terrain()
    check_programs_agree()
    print('\nok')


if __name__ == '__main__':
    main()
