"""Geometry for the blocks that are not cubes — torches, plants, doors, carpets.

The mesher's greedy path can only draw a full cube: it works on a per-face mask,
and a mask cell is one block face covering one whole square. Everything here is
the other path. A **shape** is a fixed list of quads in unit-block space, and
drawing one is copying those quads out with the block's world position added.
No merging, no face culling, no AO — a torch is six quads whatever is beside it.

That is deliberately the same data the real game uses. `referans/assets/
minecraft/models/block/*.json` describes a block as a list of boxes in a 0..16
pixel space, each box listing which of its six faces are drawn and which corner
of the texture each face takes; `box()` below is that, and every shape in this
file is a transcription of the matching model file rather than an approximation
of one. Coordinates are written in the reference's pixel units (`PX`) so the two
can be read side by side.

A quad carries:

    4 corner positions   (x, y, z) in 0..1, the block's own cell
    4 corner uvs         (u, v) in 0..1 of one atlas tile
    a face slot          which of the block's six texture names to sample
    a shading value      what the vertex format's 7th float gets

The corner *order* is not free: it is `fast_builder.emit_greedy_quad`'s, so that
the winding matches (back-face culling stays on for this pass) and so a shape
quad and a cube quad hand the same uv to the same texel. `_face_corners` and
`_auto_uv` below are that function read backwards, and `test_shapes.py` asserts
the two still agree by meshing a cube both ways.

Shapes are baked into flat numpy arrays at import (`TABLE`), because the mesher
is `@njit` and takes them as an argument — see `blocks.FACE_LAYER` for why an
argument rather than a global.
"""

import numpy as np

PX = 1.0 / 16.0

# Face order, fixed by fast_builder.emit_greedy_quad — the same six names
# blocks.py uses.
TOP, BOTTOM, FRONT, BACK, RIGHT, LEFT = range(6)

# What an unoccluded corner is worth — `fast_builder.AO_LEVELS[3]`, and the
# ceiling on the vertex format's 7th float: a cube face writes
# `face_brightness * corner_ao`, so the brightest thing in the world is a top
# face with open sky at every corner, at 1.0 * 0.8. Written out rather than
# imported because fast_builder imports *this* file; test_shapes.py pins the two
# together. Get it wrong and a carpet is 25% brighter than the floor it lies on.
AO_OPEN = 0.80

# Face brightness, emit_greedy_quad's own numbers taken through that ceiling. A
# shape quad has no AO of its own — nothing merges, and a torch has no flat face
# to occlude — so this is the whole of its shading.
FACE_SHADE = tuple(AO_OPEN * b for b in (1.0, 0.4, 0.8, 0.8, 0.6, 0.6))

# The reference marks cross-shaped and torch models `"shade": false`: no
# directional shading at all, because a flat plant lit by face normal has one
# bright side and one dark one and reads as two different plants depending on
# where you stand. Here that is the top face's brightness, which is the most any
# block gets.
UNSHADED = AO_OPEN


def _face_corners(face, x0, y0, z0, x1, y1, z1):
    """The four corners of one face of a box, in emit_greedy_quad's order."""
    if face == TOP:
        return ((x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (x1, y1, z0))
    if face == BOTTOM:
        return ((x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1))
    if face == FRONT:                                   # +Z
        return ((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1))
    if face == BACK:                                    # -Z
        return ((x1, y0, z0), (x0, y0, z0), (x0, y1, z0), (x1, y1, z0))
    if face == RIGHT:                                   # +X
        return ((x1, y0, z1), (x1, y0, z0), (x1, y1, z0), (x1, y1, z1))
    return ((x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0))     # -X


def _auto_uv(face, x0, y0, z0, x1, y1, z1):
    """The texture rect a face of this box covers, if nothing overrides it.

    Which is the reference's rule as well as ours: a face that is inset takes
    the inset part of the texture, so a cactus's side shows the middle of its
    tile and a snow layer's shows the bottom two pixels of it.
    """
    if face == TOP or face == BOTTOM:
        return (x0, z0, x1, z1)
    if face == FRONT:
        return (x0, 1.0 - y1, x1, 1.0 - y0)
    if face == BACK:
        return (1.0 - x1, 1.0 - y1, 1.0 - x0, 1.0 - y0)
    if face == RIGHT:
        return (1.0 - z1, 1.0 - y1, 1.0 - z0, 1.0 - y0)
    return (z0, 1.0 - y1, z1, 1.0 - y0)                                 # -X


def _rect_uv(face, rect):
    """A (u0, v0, u1, v1) rect spread over a face's four corners, in order.

    v runs downward from the top of the tile, which is both the reference's
    convention in its model files and what emit_greedy_quad already writes.
    """
    u0, v0, u1, v1 = rect
    if face == TOP:
        return ((u0, v0), (u0, v1), (u1, v1), (u1, v0))
    if face == BOTTOM:
        return ((u0, v0), (u1, v0), (u1, v1), (u0, v1))
    return ((u0, v1), (u1, v1), (u1, v0), (u0, v0))     # the four sides


def quad(corners, uvs, slot, shade):
    return (tuple(corners), tuple(uvs), slot, shade)


def box(x0, y0, z0, x1, y1, z1, faces=(TOP, BOTTOM, FRONT, BACK, RIGHT, LEFT),
        slot=None, uv=None, shade=None):
    """One of the reference's model elements: an axis-aligned box, some faces of.

    Bounds are in block units (write them as `n * PX` to keep the reference's
    numbers). *slot* forces every face to one texture name; the default gives
    each face its own, which is what makes a 3-texture block like a cactus come
    out with its top on top. *uv* is a per-face dict of (u0, v0, u1, v1) rects
    for the handful of faces the reference does not take straight off the box.
    """
    out = []
    for face in faces:
        rect = (uv or {}).get(face)
        if rect is None:
            rect = _auto_uv(face, x0, y0, z0, x1, y1, z1)
        out.append(quad(_face_corners(face, x0, y0, z0, x1, y1, z1),
                        _rect_uv(face, rect),
                        face if slot is None else slot,
                        FACE_SHADE[face] if shade is None else shade))
    return out


def double_sided(corners, uvs, slot=FRONT, shade=UNSHADED):
    """A plane drawn from both sides.

    Back-face culling is on for the whole pass — a chunk's see-through mesh is
    drawn with the same state as its opaque one — so a single quad would vanish
    when you walked around it. Reversing the corner order reverses the winding,
    and the uvs have to travel with their corners or the mirrored copy samples
    the tile sideways.
    """
    return [quad(corners, uvs, slot, shade),
            quad(corners[::-1], uvs[::-1], slot, shade)]


# ---------------------------------------------------------------------------
# The shapes
# ---------------------------------------------------------------------------
_PLANE_UV = ((0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0))


def _cross():
    """`block/cross` — two planes on the cell's diagonals, each double-sided.

    The reference builds it as two 14.4-wide planes rotated 45 degrees with
    `"rescale": true`, which stretches them back out to the full diagonal; the
    corners below are where that lands. Every plant in the game is this model.
    """
    a = ((0.0, 0.0, 0.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 0.0))
    b = ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 1.0), (1.0, 1.0, 0.0))
    return double_sided(a, _PLANE_UV) + double_sided(b, _PLANE_UV)


def _crop():
    """`block/crop` — four parallel planes, not a cross.

    Wheat in a row reads as a field precisely because the planes are parallel:
    a cross would give every plant the same two diagonals and the rows would
    disappear. The reference starts them a pixel below the ground (`from` y -1)
    and stops them a pixel short of the top; clamped to the cell here, because a
    quad reaching into the block below would z-fight with its top face.
    """
    out = []
    for x in (4 * PX, 12 * PX):
        corners = ((x, 0.0, 0.0), (x, 0.0, 1.0), (x, 15 * PX, 1.0), (x, 15 * PX, 0.0))
        out += double_sided(corners, _PLANE_UV)
    for z in (4 * PX, 12 * PX):
        corners = ((0.0, 0.0, z), (1.0, 0.0, z), (1.0, 15 * PX, z), (0.0, 15 * PX, z))
        out += double_sided(corners, _PLANE_UV)
    return out


def _torch():
    """`block/template_torch` — a 2x10 stick, plus two full-cell planes.

    The two planes are the whole trick and they are the reference's: the stick
    itself is only capped top and bottom by the small box, and its *sides* are
    full 16x16 quads carrying the entire (mostly transparent) tile. So the torch
    stands up straight from any angle for four quads, and the flame at the top of
    the texture is drawn where the box would have cut it off.
    """
    caps = box(7 * PX, 0.0, 7 * PX, 9 * PX, 10 * PX, 9 * PX,
               faces=(TOP, BOTTOM), slot=FRONT, shade=UNSHADED,
               uv={TOP: (7 * PX, 6 * PX, 9 * PX, 8 * PX),
                   BOTTOM: (7 * PX, 13 * PX, 9 * PX, 15 * PX)})
    ew = box(7 * PX, 0.0, 0.0, 9 * PX, 1.0, 1.0,
             faces=(RIGHT, LEFT), slot=FRONT, shade=UNSHADED)
    ns = box(0.0, 0.0, 7 * PX, 1.0, 1.0, 9 * PX,
             faces=(FRONT, BACK), slot=FRONT, shade=UNSHADED)
    return caps + ew + ns


# Which side of its own cell an oriented block's geometry hugs. The placement
# code in main.py works in the same four directions and in this order.
FACING_NAMES = ('north', 'east', 'south', 'west')        # -Z, +X, +Z, -X

_DOOR_BOUNDS = {
    # 3 pixels thick, full height, against one wall of the cell — `block/door`.
    'north': (0.0, 0.0, 0.0, 1.0, 1.0, 3 * PX),
    'south': (0.0, 0.0, 13 * PX, 1.0, 1.0, 1.0),
    'east': (13 * PX, 0.0, 0.0, 1.0, 1.0, 1.0),
    'west': (0.0, 0.0, 0.0, 3 * PX, 1.0, 1.0),
}

# The reference hangs a ladder 0.8 px off the wall it is on, which is what keeps
# it from z-fighting with the wall.
_LADDER_AT = 0.8 * PX


def _ladder(facing):
    """`block/ladder` — one plane against the wall, drawn from both sides."""
    if facing == 'north':
        z = _LADDER_AT
        corners = ((1.0, 0.0, z), (0.0, 0.0, z), (0.0, 1.0, z), (1.0, 1.0, z))
    elif facing == 'south':
        z = 1.0 - _LADDER_AT
        corners = ((0.0, 0.0, z), (1.0, 0.0, z), (1.0, 1.0, z), (0.0, 1.0, z))
    elif facing == 'east':
        x = 1.0 - _LADDER_AT
        corners = ((x, 0.0, 1.0), (x, 0.0, 0.0), (x, 1.0, 0.0), (x, 1.0, 1.0))
    else:
        x = _LADDER_AT
        corners = ((x, 0.0, 0.0), (x, 0.0, 1.0), (x, 1.0, 1.0), (x, 1.0, 0.0))
    return double_sided(corners, _PLANE_UV, shade=FACE_SHADE[FRONT])


def _cactus():
    """`block/cactus` — the sides sit flush but stop a pixel short sideways, so
    the four corner columns are open. The top and bottom are a full cube face."""
    return (box(1 * PX, 0.0, 0.0, 15 * PX, 1.0, 1.0, faces=(FRONT, BACK),
                uv={FRONT: (0.0, 0.0, 1.0, 1.0), BACK: (0.0, 0.0, 1.0, 1.0)})
            + box(0.0, 0.0, 1 * PX, 1.0, 1.0, 15 * PX, faces=(RIGHT, LEFT),
                  uv={RIGHT: (0.0, 0.0, 1.0, 1.0), LEFT: (0.0, 0.0, 1.0, 1.0)})
            + box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, faces=(TOP, BOTTOM)))


def _flat(height, inset=0.0):
    """A slab lying on the floor: carpets, pressure plates, a snow layer."""
    return box(inset, 0.0, inset, 1.0 - inset, height, 1.0 - inset)


def _lily():
    """`block/template_lily_pad` — one horizontal plane just off the ground."""
    y = 0.25 * PX
    corners = ((0.0, y, 0.0), (0.0, y, 1.0), (1.0, y, 1.0), (1.0, y, 0.0))
    uvs = ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0))
    return double_sided(corners, uvs, shade=FACE_SHADE[TOP])


# name -> quads. Order is fixed only in that `test_shapes.py` and the baked
# arrays below read it once; blocks.py refers to shapes by name.
SHAPES = {
    'cross': _cross(),
    'crop': _crop(),
    'torch': _torch(),
    'cactus': _cactus(),
    'carpet': _flat(1 * PX),
    'plate': _flat(1 * PX, inset=1 * PX),
    'snow_layer': _flat(2 * PX),
    'lily': _lily(),
}
for _f in FACING_NAMES:
    SHAPES['door_' + _f] = box(*_DOOR_BOUNDS[_f], slot=FRONT)
    SHAPES['ladder_' + _f] = _ladder(_f)

# Shapes the player walks straight through, which `blocks.COLLIDES` reads.
#
# Two kinds are in here. Plants, a torch and a ladder have no business stopping
# anyone. The flat ones are subtler: a carpet occupies the bottom pixel of its
# own cell, so the floor the player already stands on *is* its underside — give
# it collision and they stand a whole block above the rug.
#
# That is also why there is no slab and no stair in this file. Their geometry
# would be two more `box()` calls, but they are half a block tall, and
# `engine/camera.py` asks one question about a cell — solid or not. Standing on
# a slab needs a collision box shorter than its cell, which nothing there can
# express yet.
WALKTHROUGH = frozenset({'cross', 'crop', 'torch', 'carpet', 'plate',
                         'snow_layer', 'lily'}
                        | {'ladder_' + f for f in FACING_NAMES})

# The reference nudges cross-shaped models up to a quarter of a block sideways
# (`offset_type: XZ` on their block state), from a hash of the block position.
# Without it a meadow is a grid, and a grid is the one thing a meadow is not.
JITTERED = frozenset(('cross', 'crop'))

# How far, in blocks. The reference's is +/-0.25 and it applies to the whole
# 45-degree diagonal, which already reaches the cell corners, so a plant leans a
# little into its neighbour — which is the point.
JITTER = 0.25


# ---------------------------------------------------------------------------
# Baked, for the mesher
# ---------------------------------------------------------------------------
def _bake():
    """SHAPES as flat arrays: index 0 is the cube, which has no quads here."""
    names = ['<cube>'] + sorted(SHAPES)
    start = [0]
    pos, uv, slot, shade = [], [], [], []
    for name in names[1:]:
        for corners, uvs, face_slot, face_shade in SHAPES[name]:
            pos.append(corners)
            uv.append(uvs)
            slot.append(face_slot)
            shade.append(face_shade)
        start.append(len(pos))
    return (names,
            np.array([0] + start, dtype=np.int32),
            np.array(pos, dtype=np.float32).reshape(-1, 4, 3),
            np.array(uv, dtype=np.float32).reshape(-1, 4, 2),
            np.array(slot, dtype=np.int32),
            np.array(shade, dtype=np.float32),
            np.array([name in JITTERED for name in names], dtype=np.uint8))


NAMES, START, POS, UV, SLOT, SHADE, JITTER_FLAG = _bake()

# name -> its index in the baked arrays. 0 is the cube: blocks.SHAPE_OF hands
# the mesher a 0 for every ordinary block, a -1 for the ones that are not drawn
# at all (air, water), and one of these otherwise.
INDEX = {name: i for i, name in enumerate(NAMES)}

# The most quads any one shape needs, which is what bounds the overflow check in
# build_chunk_mesh_fast.
MAX_QUADS = int(np.diff(START[1:]).max())

assert START[0] == 0 and START[1] == 0, 'shape 0 is the cube and draws nothing'
assert len(START) == len(NAMES) + 1
assert POS.min() >= 0.0 and POS.max() <= 1.0, 'a shape reaches outside its block'
assert UV.min() >= 0.0 and UV.max() <= 1.0, 'a shape samples outside its tile'
