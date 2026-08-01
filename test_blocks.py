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

from engine import hud as hudmod
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


def check_cells(hud, height, what):
    """Every cell in the list is hittable, at every scroll position.

    Cells are laid out unscrolled and moved by a shader uniform, so the hit test
    has to subtract the scroll back off — and it has to refuse cells that have
    scrolled out of the viewport, which are still in the list but not on screen.
    """
    for scroll in (0.0, hud._max_scroll / 3, hud._max_scroll):
        hud.set_scroll(scroll)
        seen = 0
        for i, (block_id, (x0, y0, x1, y1), _) in enumerate(hud._inv_cells):
            # hit_test takes pygame's y, measured from the top.
            mx = int((x0 + x1) / 2)
            my = height - int((y0 + y1) / 2 + hud.scroll)
            hit = hud.hit_test(mx, my)
            vy0, vy1 = hud._viewport[1], hud._viewport[3]
            inside = vy0 <= (y0 + y1) / 2 + hud.scroll <= vy1
            if inside:
                assert hit == i, f'{what}: cell {i} not hittable at scroll {scroll:.0f}'
                assert hud.block_at(i) == block_id
                seen += 1
            else:
                assert hit != i, f'{what}: cell {i} hittable while scrolled out'
        assert seen, f'{what}: nothing visible at scroll {scroll:.0f}'
    hud.set_scroll(0.0)


def check_panel_fits(hud, width, height, what):
    px0, py0, px1, py1 = hud._panel_px
    assert 0 <= px0 and px1 <= width, f'{what}: panel off screen horizontally'
    assert py1 <= height, f'{what}: panel above the top of the window'
    assert py0 >= hud.BOTTOM_PY + hud.SLOT_PX, f'{what}: panel overlaps the hotbar'
    for i, (x0, y0, x1, y1) in enumerate(hud._tabs):
        assert px0 <= x0 and x1 <= px1 and py0 <= y0 and y1 <= py1, \
            f'{what}: tab {i} outside the panel'
        assert hud.tab_at(int((x0 + x1) / 2), height - int((y0 + y1) / 2)) == i, \
            f'{what}: tab {i} not clickable'


def check_picker(width, height):
    ctx = moderngl.create_standalone_context()
    hud = HUDRenderer(ctx, width, height)

    check_panel_fits(hud, width, height, f'{width}x{height}')
    panel = hud._panel_px

    # Every tab lists its own category, in registry order, and scrolls.
    for i, (name, ids) in enumerate(hudmod.TABS):
        hud.set_tab(i)
        assert [c[0] for c in hud._inv_cells] == list(ids), f'tab {name!r} lists the wrong blocks'
        check_cells(hud, height, f'tab {name!r}')
        assert hud._panel_px == panel, 'the panel resized when the tab changed'

    hud.set_tab(0)
    assert hud.hit_test(-5, -5) == -1
    assert hud.block_at(-1) is None and hud.block_at(len(blocks.CREATIVE)) is None

    # A slot swap has to survive the geometry rebuild it triggers.
    hud.set_slot(0, blocks.CREATIVE[-1])
    assert hud.hotbar[0] == blocks.CREATIVE[-1]

    ctx.release()
    print(f'{len(hudmod.TABS)} tabs, all cells clickable at every scroll, {width}x{height}')


def check_search(width, height):
    """Search spans every category, and the panel does not move while typing."""
    ctx = moderngl.create_standalone_context()
    hud = HUDRenderer(ctx, width, height)
    hud.set_tab(7)                       # a search must escape the current tab
    panel = hud._panel_px

    for query in ('wool', 'deepslate', 'oak log', 'e', 'CONCRETE', 'a'):
        hud.set_query(query)
        want = [b for b in blocks.CREATIVE
                if query.strip().lower() in blocks.BLOCK_NAMES[b].lower()]
        got = [c[0] for c in hud._inv_cells]
        assert got == want, f'{query!r}: {len(got)} hits, expected {len(want)}'
        assert got, f'{query!r} matched nothing — pick a query that does'
        assert hud._panel_px == panel, f'{query!r}: the panel resized while typing'
        check_cells(hud, height, f'query {query!r}')

    # No match must not divide by zero or leave the last query's cells behind.
    hud.set_query('zzzznope')
    assert hud._inv_cells == [] and hud._max_scroll == 0
    assert hud._panel_px == panel, 'the panel resized on an empty result'

    hud.set_query('')
    assert len(hud._inv_cells) == len(blocks.CREATIVE), 'clearing did not restore'

    ctx.release()
    print(f'search spans all tabs, panel fixed at {width}x{height}')


def main():
    check_registry()
    check_atlas_matches_registry()
    for size in ((1200, 800), (1000, 700), (640, 480)):
        check_picker(*size)
    check_search(1200, 800)
    check_search(640, 480)
    print('\nok')


if __name__ == '__main__':
    main()
