"""Block registry + creative grid self-check.

Guards two things that fail as "looks wrong" or "clicks wrong" rather than as
an exception:

  * `texture.png` matches `world/blocks.py`. Adding a block without rerunning
    `python build_atlas.py` leaves FACE_LAYER pointing past the end of the
    texture array, and OpenGL clamps rather than complains — the new block
    silently wears the last texture in the atlas.
  * every creative cell can be clicked. The grid is laid out in pixels and hit
    tested in pixels with the y axis flipped between them, so an off-by-one
    there hands the player a different block than the one under the cursor.

Run: python test_blocks.py
"""

import moderngl
import pygame as pg

from engine.hud import HOTBAR_SLOTS, HUDRenderer
from world import blocks

ATLAS = 'texture.png'
TILE = 16


def check_registry():
    for bid, faces in blocks.BLOCK_FACES.items():
        assert len(faces) == 6, f'{bid}: {len(faces)} faces'
        for face, name in enumerate(faces):
            assert blocks.TEXTURES[blocks.FACE_LAYER[bid, face]] == name, \
                f'{blocks.BLOCK_NAMES[bid]} face {face} points at the wrong layer'

    assert blocks.FACE_LAYER.max() < blocks.LAYER_COUNT
    assert set(blocks.CREATIVE) == set(blocks.BLOCK_NAMES), 'a block is unreachable'
    assert len(blocks.CREATIVE) == len(set(blocks.CREATIVE)), 'duplicate in CREATIVE'
    assert all(b in blocks.BLOCK_NAMES for b in blocks.HOTBAR_DEFAULT)
    assert len(blocks.HOTBAR_DEFAULT) == HOTBAR_SLOTS
    print(f'registry: {len(blocks.BLOCK_NAMES)} blocks, '
          f'{blocks.LAYER_COUNT} atlas layers')


def check_atlas_matches_registry():
    width, height = pg.image.load(ATLAS).get_size()
    assert width == TILE, f'{ATLAS} is {width}px wide, expected {TILE}'
    assert height == TILE * blocks.LAYER_COUNT, (
        f'{ATLAS} holds {height // TILE} layers, the registry wants '
        f'{blocks.LAYER_COUNT} — rerun `python build_atlas.py`')
    print(f'{ATLAS}: {width}x{height}, {height // TILE} layers')


def check_every_cell_is_clickable(width, height):
    ctx = moderngl.create_standalone_context()
    hud = HUDRenderer(ctx, width, height)

    for i, (block_id, (x0, y0, x1, y1), _) in enumerate(hud._inv_cells):
        # hit_test takes pygame's y, measured from the top.
        mx = int((x0 + x1) / 2)
        my = height - int((y0 + y1) / 2)
        assert hud.hit_test(mx, my) == i, f'cell {i} not hittable at its centre'
        assert hud.block_at(i) == block_id

    assert hud.hit_test(-5, -5) == -1
    assert hud.block_at(-1) is None and hud.block_at(len(blocks.CREATIVE)) is None

    # A slot swap has to survive the geometry rebuild it triggers.
    hud.set_slot(0, blocks.CREATIVE[-1])
    assert hud.hotbar[0] == blocks.CREATIVE[-1]

    ctx.release()
    print(f'{len(hud._inv_cells)} creative cells clickable at {width}x{height}')


def main():
    check_registry()
    check_atlas_matches_registry()
    for size in ((1200, 800), (1000, 700), (640, 480)):
        check_every_cell_is_clickable(*size)
    print('\nok')


if __name__ == '__main__':
    main()
