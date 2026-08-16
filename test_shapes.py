"""Self-check for the blocks that are not cubes. Run: python test_shapes.py

Torches, plants, doors and carpets are drawn by a second path in the mesher
(`emit_shape_quads`) that has no greedy merging, no face culling and no AO. Not
one of its failure modes raises: a shape whose corner order drifts from
`emit_greedy_quad`'s comes out inside-out or with its texture rotated, a plant
that lands in the opaque buffer is drawn as a solid box, a door with no facings
faces one way forever, and a carpet with the wrong shading is a visibly
brighter square on the floor.

The load-bearing assert is the first one. `world/shapes.py` was written by
reading `emit_greedy_quad` backwards — same corner order, same uv rule, same
brightness — so a full-size `box()` has to come out byte for byte identical to
the quad the greedy mesher writes for the same cube face. Everything else in
the file is built out of `box()`, so if that holds, the rest is geometry.

Nothing here needs a window: the mesher is plain numpy.
"""

import numpy as np

from world import blocks, shapes
from world.blocks import (BLOCK_NAMES, COLLIDES, FACE_LAYER, FACING, OPAQUE,
                          SHAPE_NAME, SHAPE_OF, SHAPE_TABLE, WALL_MOUNTED)
from world.fast_builder import (AO_FULL, AO_LEVELS, CHUNK_HEIGHT, CHUNK_SIZE,
                                MAX_FACES_ALPHA, NO_NEIGHBOR,
                                build_chunk_mesh_fast, emit_greedy_quad,
                                make_mesh_buffers)

BUFFERS = make_mesh_buffers()
EMPTY = (NO_NEIGHBOR,) * 4

STONE = 3
GRASS_PLANT = next(b for b, s in SHAPE_NAME.items() if s == 'cross')
TORCH = next(b for b, s in SHAPE_NAME.items() if s == 'torch')
CARPET = next(b for b, s in SHAPE_NAME.items() if s == 'carpet')
CACTUS = next(b for b, s in SHAPE_NAME.items() if s == 'cactus')
DOOR = min(b for b in FACING if SHAPE_NAME.get(b, '').startswith('door'))
LADDER = min(WALL_MOUNTED)
FURNACE = 94                      # a cube with four facings, not a shape


def world():
    return np.zeros((CHUNK_SIZE, CHUNK_HEIGHT, CHUNK_SIZE), dtype=blocks.BLOCK_DTYPE)


def mesh(grid, lod=0):
    """(opaque vertices, see-through vertices) as (n, 4, 7) quad arrays."""
    v, _, tv, _ = build_chunk_mesh_fast(grid, 0, 0, EMPTY, *BUFFERS,
                                        FACE_LAYER, OPAQUE, SHAPE_TABLE, lod)
    return v.reshape(-1, 4, 7), tv.reshape(-1, 4, 7)


def shape_quads(name):
    """One shape's quads, as the mesher would write them at the origin."""
    grid = world()
    bid = next(b for b, s in SHAPE_NAME.items() if s == name)
    grid[0, 0, 0] = bid
    return mesh(grid)[1], bid


def check_a_full_box_is_a_cube_face():
    """`box(0,0,0,1,1,1)` has to equal what the greedy mesher writes.

    Corner order, uv rule and brightness all live twice — once in
    emit_greedy_quad, once in world/shapes.py — and nothing else can catch them
    drifting apart. A rotated corner order turns every torch inside out; a
    flipped v turns every door upside down.
    """
    cube = shapes.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    assert len(cube) == 6

    for face, (corners, uvs, slot, shade) in enumerate(cube):
        assert slot == face, 'a full box must sample each face with its own texture'
        want = np.zeros(4 * 7, dtype=np.float32)
        emit_greedy_quad(want, 0, 0, 0, 0, 0, 0, 1, 1, face, STONE, AO_FULL,
                         FACE_LAYER)
        want = want.reshape(4, 7)

        got = np.array([[*corners[i], uvs[i][0], uvs[i][1],
                         FACE_LAYER[STONE, face], shade] for i in range(4)],
                       dtype=np.float32)
        assert np.array_equal(got, want), (
            f'face {face} disagrees with emit_greedy_quad\n{got}\n{want}')

    assert shapes.AO_OPEN == AO_LEVELS[3], \
        'shapes.AO_OPEN drifted from fast_builder.AO_LEVELS — shapes would not ' \
        'be as bright as the terrain around them'
    print('a full box matches emit_greedy_quad on all 6 faces, uvs and shading')


def check_quad_counts_match_the_models():
    """The reference's own element counts, so a shape cannot quietly lose a face.

    A cross is two planes drawn from both sides; a torch is the reference's two
    caps plus four full-cell planes; a crop is four parallel planes, not a cross.
    """
    want = {'cross': 4, 'crop': 8, 'torch': 6, 'cactus': 6, 'carpet': 6,
            'plate': 6, 'snow_layer': 6, 'lily': 2,
            'door_north': 6, 'ladder_north': 2,
            # One entry per drawn face of the matching model file, counted off
            # its `elements`: template_torch_wall is one box with all six,
            # template_lantern is 6 + 5 + 2 + 2, flower_pot is 6 + 6 + 4 + 4 + 2
            # and the plant in it is another cross.
            'torch_wall_west': 6, 'pot_empty': 22, 'pot': 26, 'lantern': 15,
            'chain': 4, 'end_rod': 11, 'lightning_rod': 11, 'fan': 8,
            'spore_blossom': 10, 'sea_pickle': 11, 'egg': 6, 'cake': 6,
            'composter': 18, 'anvil': 21, 'enchanting_table': 6,
            'daylight_detector': 6, 'lectern_north': 16,
            'stonecutter_north': 8}
    for name, count in want.items():
        assert len(shapes.SHAPES[name]) == count, \
            f'{name}: {len(shapes.SHAPES[name])} quads, expected {count}'

    for name, quads in shapes.SHAPES.items():
        for corners, uvs, slot, shade in quads:
            assert 0 <= slot < 6, f'{name}: face slot {slot}'
            area = np.cross(np.subtract(corners[1], corners[0]),
                            np.subtract(corners[3], corners[0]))
            assert np.linalg.norm(area) > 1e-6, f'{name}: a degenerate quad'
    print(f'{len(shapes.SHAPES)} shapes, {len(shapes.POS)} quads, '
          f'counts match the reference models')


def check_a_cactus_is_a_full_width_block():
    """Every side spans the cell corner to corner and stands a pixel back from it.

    The reference gets that from *which element carries which faces*: north and
    south come off the box inset in z, east and west off the one inset in x.
    Swap them and nothing here changes except the geometry — six quads, valid
    winding, whole-tile uvs — but each side is 14 pixels wide and flush, so a
    column of cacti has a 1-pixel slit down all four of its corners and the side
    texture is squeezed into 7/8 of the width it was drawn for.
    """
    px = shapes.PX
    want = {shapes.FRONT: (0.0, 1.0, 15 * px, 15 * px),      # +Z, set back in z
            shapes.BACK: (0.0, 1.0, 1 * px, 1 * px),         # -Z
            shapes.RIGHT: (15 * px, 15 * px, 0.0, 1.0),      # +X, set back in x
            shapes.LEFT: (1 * px, 1 * px, 0.0, 1.0),         # -X
            shapes.TOP: (0.0, 1.0, 0.0, 1.0),                # a full cube face
            shapes.BOTTOM: (0.0, 1.0, 0.0, 1.0)}
    for corners, uvs, slot, _ in shapes.SHAPES['cactus']:
        xs = [c[0] for c in corners]
        zs = [c[2] for c in corners]
        assert (min(xs), max(xs), min(zs), max(zs)) == want[slot], \
            f'cactus face {slot} is at {min(xs), max(xs), min(zs), max(zs)}, ' \
            f'expected {want[slot]}'
        us = [u for u, _v in uvs]
        vs = [_v for _u, _v in uvs]
        assert (min(us), max(us), min(vs), max(vs)) == (0.0, 1.0, 0.0, 1.0), \
            f'cactus face {slot} samples part of its tile, not all of it'
    print('a cactus is a full-width block with each side set one pixel in')


def check_shapes_go_to_the_blended_pass():
    """Every one of them, and none of the opaque one.

    They all carry alpha, and the opaque terrain pass has no discard on purpose
    — a plant that landed in it would be drawn as a solid box with the holes
    filled in, and it would cost the whole world its early-Z to fix there.
    """
    for bid, name in SHAPE_NAME.items():
        grid = world()
        grid[8, 40, 8] = bid
        opaque, alpha = mesh(grid)
        n = shapes.START[SHAPE_OF[bid] + 1] - shapes.START[SHAPE_OF[bid]]
        assert len(opaque) == 0, f'{BLOCK_NAMES[bid]} put geometry in the opaque pass'
        assert len(alpha) == n, f'{BLOCK_NAMES[bid]}: {len(alpha)} quads, expected {n}'
    print(f'all {len(SHAPE_NAME)} shape blocks mesh into the see-through buffer')


def check_a_plant_hides_nothing():
    """The floor under a tuft of grass keeps its top face, and its shading.

    Both halves matter and they are different tables: OPAQUE says whether the
    face behind it is drawn at all, and get_face_ao reads the same table to
    decide whether it casts a contact shadow. A solid block is the control —
    without it "nothing changed" could just mean nothing was placed.
    """
    floor = world()
    floor[:, 9, :] = STONE
    bare, _ = mesh(floor)

    floor[4, 10, 4] = GRASS_PLANT
    grassy, plants = mesh(floor)
    assert np.array_equal(bare, grassy), 'a plant hid or darkened the floor under it'
    assert len(plants) == 4, 'the plant itself went missing'

    floor[4, 10, 4] = TORCH
    lit, _ = mesh(floor)
    assert np.array_equal(bare, lit), 'a torch darkened the floor under it'

    floor[4, 10, 4] = STONE
    stacked, _ = mesh(floor)
    assert not np.array_equal(bare, stacked), \
        'a solid block is supposed to occlude — this check proves nothing otherwise'
    print('grass and torches leave the floor exactly as it was')


def check_the_greedy_path_never_sees_them():
    """A chunk of nothing but plants produces no cube geometry at all."""
    grid = world()
    grid[:, 40, :] = GRASS_PLANT
    opaque, alpha = mesh(grid)
    assert len(opaque) == 0, 'a plant reached the greedy mesher'
    assert len(alpha) == CHUNK_SIZE * CHUNK_SIZE * 4


def check_the_jitter_is_positional():
    """The same tuft lands in the same place; two tufts do not.

    The offset is the reference's `offset_type: XZ` and it is hashed from the
    block's *world* position, so a chunk rebuild has to put the plant back where
    it was — and it must stay inside a quarter block, or a plant leans into the
    next cell far enough to look rooted in it.
    """
    grid = world()
    grid[3, 40, 3] = grid[9, 40, 9] = GRASS_PLANT
    first = mesh(grid)[1]
    assert np.array_equal(first, mesh(grid)[1]), 'the jitter is not deterministic'

    a = first[:4, :, [0, 2]] - np.array([3.0, 3.0], dtype=np.float32)
    b = first[4:, :, [0, 2]] - np.array([9.0, 9.0], dtype=np.float32)
    assert not np.allclose(a, b), 'two plants got the same offset — a grid, not a meadow'
    for offsets in (a, b):
        reach = np.abs(offsets - np.clip(offsets, 0.0, 1.0)).max()
        assert reach <= shapes.JITTER + 1e-6, f'a plant reached {reach} outside its cell'

    # Nothing else may move: a door has to line up with the frame it is in.
    grid = world()
    grid[3, 40, 3] = DOOR
    door = mesh(grid)[1]
    assert door[..., 0].min() == 3.0 and door[..., 2].min() == 3.0, \
        'a door was jittered off its own cell'
    print(f'plants are offset up to {shapes.JITTER} blocks from a hash of where they are')


def check_doors_face_four_ways():
    """Four ids, four different geometries, each hugging the side it is named for.

    Facing is which wall of its own cell the slab sits against — that is the
    contract main.orient places by, and if two facings came out the same, half
    the doors in the world would open through the wall beside them.
    """
    axis = {0: (2, 'min'), 1: (0, 'max'), 2: (2, 'max'), 3: (0, 'min')}
    seen = []
    for facing, bid in enumerate(FACING[DOOR]):
        grid = world()
        grid[5, 40, 5] = bid
        quads = mesh(grid)[1]
        assert len(quads) == 6, f'facing {facing}: {len(quads)} quads'

        component, side = axis[facing]
        cell = 5.0
        span = quads[..., component]
        if side == 'min':
            assert span.min() == cell and span.max() <= cell + 0.25, \
                f'facing {facing} is not against the low wall'
        else:
            assert span.max() == cell + 1.0 and span.min() >= cell + 0.75, \
                f'facing {facing} is not against the high wall'
        seen.append(quads.tobytes())

    assert len(set(seen)) == 4, 'two door facings came out identical'

    # And the ladder family, which main.orient places against the wall that was
    # clicked rather than by yaw — same four ids, same contract.
    assert len(FACING[LADDER]) == 4
    assert len({mesh(_one(bid))[1].tobytes() for bid in FACING[LADDER]}) == 4, \
        'two ladder facings came out identical'
    print('doors and ladders: 4 facings each, every one against its own wall')


def _one(bid):
    grid = world()
    grid[5, 40, 5] = bid
    return grid


def check_collision_matches_the_shape():
    """`blocks.COLLIDES` is the only thing engine/camera.py asks about a block."""
    for bid, name in SHAPE_NAME.items():
        want = name not in shapes.WALKTHROUGH
        assert bool(COLLIDES[bid]) == want, \
            f'{BLOCK_NAMES[bid]} ({name}) is on the wrong side of COLLIDES'

    assert not COLLIDES[0] and not COLLIDES[blocks.WATER]
    assert not COLLIDES[GRASS_PLANT] and not COLLIDES[TORCH] and not COLLIDES[CARPET]
    assert COLLIDES[DOOR] and COLLIDES[CACTUS] and COLLIDES[STONE]
    walk = int((~COLLIDES).sum())
    print(f'{walk} block ids you walk through, {int(COLLIDES.sum())} that stop you')


def check_overflow_drops_plants_not_terrain():
    """A chunk packed solid with plants stops at the cap and keeps its terrain.

    The see-through buffer is the small one and it is shared with the glass, so
    it is the one that gives up — the same rule the transparent pass already
    had, now reachable by generated terrain rather than only by a player.
    """
    grid = world()
    grid[:, :20, :] = STONE
    grid[:, 20:120, :] = GRASS_PLANT
    opaque, alpha = mesh(grid)
    assert len(alpha) <= MAX_FACES_ALPHA, 'the cap did not engage'
    assert len(alpha) > MAX_FACES_ALPHA - 4, 'the cap engaged far too early'
    assert len(opaque) > 0, 'the terrain went missing when the plant buffer filled'
    print(f'overflow: {len(alpha)} shape quads capped, {len(opaque)} opaque quads kept')


def check_the_far_lod_drops_them():
    """Level 2 gives up ground cover; level 1 must not."""
    grid = world()
    grid[:, 9, :] = STONE
    grid[4, 10, 4] = GRASS_PLANT
    counts = [len(mesh(grid, lod)[1]) for lod in (0, 1, 2)]
    assert counts[0] == counts[1] == 4, f'level 1 changed the ground cover: {counts}'
    assert counts[2] == 0, f'level 2 kept it: {counts}'
    print('lod: ground cover survives level 1 and is dropped at level 2')


def check_a_furnace_turns_without_becoming_a_shape():
    """Four ids of one cube, its front texture on a different side in each.

    `blocks._oriented` is the whole of the furnace's facing, and the point of
    doing it in the texture table rather than in world/shapes.py is that nothing
    else moves: the variants stay cubes, so they greedy-merge, keep early-Z and
    stay out of the blended pass. Written wrong, a furnace ends up with two
    doors or none.
    """
    front = FACE_LAYER[FURNACE, blocks.FRONT]
    side = FACE_LAYER[FURNACE, blocks.BACK]
    assert front != side, 'the furnace row has no front face to turn'

    looks_out_of = (blocks.BACK, blocks.RIGHT, blocks.FRONT, blocks.LEFT)
    for facing, bid in enumerate(FACING[FURNACE]):
        assert SHAPE_OF[bid] == 0, 'an oriented cube stopped being a cube'
        row = FACE_LAYER[bid]
        assert row[looks_out_of[facing]] == front, \
            f'facing {facing} does not carry its front'
        for f in (blocks.FRONT, blocks.BACK, blocks.RIGHT, blocks.LEFT):
            if f != looks_out_of[facing]:
                assert row[f] == side, f'facing {facing} has a second front'
        assert row[blocks.TOP] == FACE_LAYER[FURNACE, blocks.TOP]
        assert row[blocks.BOTTOM] == FACE_LAYER[FURNACE, blocks.BOTTOM]

    grid = world()
    grid[5, 40, 5] = FACING[FURNACE][1]
    opaque, alpha = mesh(grid)
    assert len(opaque) == 6 and len(alpha) == 0, \
        'an oriented cube left the greedy path'
    print(f'{BLOCK_NAMES[FURNACE].lower()}: {len(FACING[FURNACE])} facings, '
          'still one greedy cube')


def main():
    check_a_full_box_is_a_cube_face()
    check_quad_counts_match_the_models()
    check_a_cactus_is_a_full_width_block()
    check_shapes_go_to_the_blended_pass()
    check_a_plant_hides_nothing()
    check_the_greedy_path_never_sees_them()
    check_the_jitter_is_positional()
    check_doors_face_four_ways()
    check_a_furnace_turns_without_becoming_a_shape()
    check_collision_matches_the_shape()
    check_overflow_drops_plants_not_terrain()
    check_the_far_lod_drops_them()
    print('\nok')


if __name__ == '__main__':
    main()
