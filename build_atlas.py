"""Bake `texture.png` from the reference game's block textures.

    python build_atlas.py

Reads every texture named in `world/blocks.py` out of
`referans/assets/minecraft/textures/block/` and writes them as one 16-wide
vertical strip — layer *i* is the *i*-th tile down, which is exactly what
`ModernGLRenderer.create_texture_array(path, 1, LAYER_COUNT)` unpacks.

Run this after editing the block table in `world/blocks.py`, then commit the
new `texture.png`. `referans/` is gitignored, so the game must never read it at
runtime; the baked atlas is the only thing it sees. Needs Pillow, which the
game itself does not — like `referans/`, it is a build-step prerequisite and
deliberately not in `requirements.txt`.

Three things happen on the way in, and all three exist so the mesher does not
have to change:

* **Animated textures are cropped to their first frame.** A few files
  (`sea_lantern`, `magma`) ship as a vertical film strip, 16x80 rather than
  16x16.
* **Greyscale masters are tinted** (`blocks.TINTS`). The real game tints grass
  and leaves per biome from a colormap; we have no biomes, so it happens once,
  here, with the plains colours.
* **Alpha is flattened** against the average of the texture's own opaque
  pixels — for every block outside `blocks.TRANSPARENT`. Leaves are 30-44%
  holes, and `chunk.frag` has no alpha test on purpose: `discard` costs early-Z
  for the entire terrain pass, and near-to-far chunk sorting is worth ~20% of
  frame time precisely because early-Z works. Filling the holes with the leaf's
  own green keeps the pass opaque and reads as dense foliage.

  The blocks in `blocks.TRANSPARENT` keep their alpha instead, and are drawn by
  a second, blended pass (`chunk_alpha.frag`) that the opaque pass never sees.
  Their fully-clear texels still get their colour filled in, because mipmapping
  averages RGB across them and glass ships those texels black — untouched, a
  distant window grows a dark rim.
"""

import os
import sys

import numpy as np
import pygame as pg
from PIL import Image

from world import blocks

SRC = os.path.join('referans', 'assets', 'minecraft', 'textures', 'block')
OUT = 'texture.png'
TILE = 16

# The textures that keep their alpha: the see-through cubes plus every non-cube
# shape. A flower is 88% holes and a torch 92% — for those this is not a design
# choice the way it is for glass, it is the only way they are a torch at all.
ALPHA_TEXTURES = {name for bid in blocks.CUTOUT
                  for name in blocks.BLOCK_FACES[bid]}
_CUBE_TEXTURES = {name for bid, faces in blocks.BLOCK_FACES.items()
                  if bid not in blocks.CUTOUT for name in faces}

# A layer is one thing or the other, so a texture used by both kinds of block
# can only be baked once — and it is baked with alpha kept. That is harmless
# exactly when the file has no transparency to keep, which is the case for every
# sharing there is now (a carpet and its wool block, a snow layer and its snow).
# Checked against the files in main(), not asserted away here: the alternative
# is a second copy of the same 16x16 in the atlas for nothing.
SHARED_TEXTURES = ALPHA_TEXTURES & _CUBE_TEXTURES


def load_tile(name):
    """One texture as a float32 (TILE, TILE, 4) RGBA array, tint applied.

    Read with Pillow, not pygame. SDL_image decodes a greyscale-plus-
    transparency PNG into a 16-bit RGB565 surface and gets the colours wrong:
    `spruce_leaves.png` is the one texture in the reference shipped that way,
    and it came back as a yellow ramp (255, 242, 213 where the file says
    154, 154, 154). SPRUCE_TINT then multiplied that into the garish green the
    game was drawing. Every other texture is palettes or RGB and both readers
    agree on them exactly.
    """
    path = os.path.join(SRC, name + '.png')
    if not os.path.exists(path):
        raise SystemExit(f'missing texture: {path}')

    image = Image.open(path)
    width, height = image.size
    if height > width:
        # Film strip: take the first frame only.
        image = image.crop((0, 0, width, width))
    if image.size != (TILE, TILE):
        raise SystemExit(f'{name}: expected {TILE}x{TILE}, got {image.size}')

    # Pillow is (row, column); everything downstream is pygame's (x, y).
    rgba = np.asarray(image.convert('RGBA'), np.float32).transpose(1, 0, 2).copy()

    tint = blocks.TINTS.get(name)
    if tint:
        rgba[..., :3] *= np.float32(tint) / 255.0

    return rgba


def over(top, bottom):
    """Standard source-over composite of two RGBA tiles."""
    alpha = top[..., 3:] / 255.0
    out = bottom.copy()
    out[..., :3] = top[..., :3] * alpha + bottom[..., :3] * (1.0 - alpha)
    out[..., 3:] = np.maximum(top[..., 3:], bottom[..., 3:])
    return out


def _own_colour(rgba, visible):
    """Mean RGB of the texels that are actually there, for filling in the rest."""
    if not visible.any():
        return np.zeros(3, np.float32)
    return rgba[visible][:, :3].mean(axis=0)


def flatten(rgba):
    """Drop alpha by compositing over the mean of the tile's own opaque pixels."""
    alpha = rgba[..., 3:] / 255.0
    fill = _own_colour(rgba, alpha[..., 0] > 0.5)
    out = np.empty_like(rgba)
    out[..., :3] = rgba[..., :3] * alpha + fill * (1.0 - alpha)
    out[..., 3] = 255.0
    return out


def bleed(rgba):
    """Keep alpha; only repaint the texels that are completely clear.

    Those texels have no colour of their own — glass ships them black — and the
    mip chain averages RGB regardless of alpha, so left black they bleed a dark
    rim into every reduced level. Anything with partial alpha is left exactly as
    it is: stained glass is a uniform 40-64% across the tile, and pulling it
    toward a "solid" average would drain the colour out of it.
    """
    clear = rgba[..., 3] < 8
    if not clear.any():
        return rgba
    out = rgba.copy()
    out[clear, :3] = _own_colour(rgba, ~clear)
    return out


def build_tile(name):
    if name in blocks.COMPOSITES:
        base, overlay = blocks.COMPOSITES[name]
        rgba = over(load_tile(overlay), load_tile(base))
    else:
        rgba = load_tile(name)
    return bleed(rgba) if name in ALPHA_TEXTURES else flatten(rgba)


def main():
    if not os.path.isdir(SRC):
        raise SystemExit(f'{SRC} not found — run from the repo root, with the '
                         'reference jar extracted into referans/')

    pg.init()
    pg.display.set_mode((1, 1))   # convert_alpha() needs a display format

    for name in sorted(SHARED_TEXTURES):
        tile = build_tile(name)
        assert tile[..., 3].min() == 255, \
            (f'{name} is used by a cube block and by a see-through one, and it '
             'has transparency — the cube would be drawn with holes in it')

    n = blocks.LAYER_COUNT
    # surfarray is (width, height): one column of n tiles.
    atlas = np.zeros((TILE, TILE * n, 4), dtype=np.float32)
    for i, name in enumerate(blocks.TEXTURES):
        atlas[:, i * TILE:(i + 1) * TILE, :] = build_tile(name)

    atlas = np.clip(atlas, 0, 255).astype(np.uint8)
    # RGBA, so the see-through layers survive the round trip. Every other layer
    # is alpha 255 and renders exactly as it did when this file was RGB.
    surface = pg.Surface((TILE, TILE * n), pg.SRCALPHA, 32)
    pg.surfarray.blit_array(surface, atlas[..., :3])
    alpha_view = pg.surfarray.pixels_alpha(surface)
    alpha_view[:] = atlas[..., 3]
    del alpha_view                      # unlock before saving
    pg.image.save(surface, OUT)

    # Round-trip check: the written file must unpack the way create_texture_array
    # will unpack it. Off-by-one in the layer order repaints the whole world and
    # nothing would raise — it would just look wrong.
    written = pg.image.load(OUT).convert_alpha()
    assert written.get_size() == (TILE, TILE * n), written.get_size()
    for probe in ('dirt', 'stone', 'grass_block_top', 'glass', 'white_stained_glass'):
        layer = blocks.TEXTURES.index(probe)
        sub = written.subsurface(pg.Rect(0, layer * TILE, TILE, TILE))
        tile = np.dstack((pg.surfarray.array3d(sub), pg.surfarray.array_alpha(sub)))
        want = np.clip(build_tile(probe), 0, 255).astype(np.uint8)
        assert np.array_equal(tile, want), f'{probe} landed on the wrong layer'

    print(f'{OUT}: {TILE}x{TILE * n}, {n} layers, '
          f'{len(blocks.BLOCK_NAMES)} block types')
    print('unpacked by ModernGLRenderer.create_texture_array(path, 1, '
          f'{n}) — blocks.LAYER_COUNT')


if __name__ == '__main__':
    sys.exit(main())
