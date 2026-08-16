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

import math

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


def spin(quads, origin, axis, angle, rescale=False):
    """A model element's `rotation`, applied to quads already built.

    The reference describes a box as axis-aligned plus one optional rotation
    about one axis through one point, and a handful of shapes here need the
    second half: a wall torch leans 22.5 degrees off its wall, a lantern's chain
    and a coral fan's blades are turned 45 and 22.5. Bounds are in block units
    like `box`'s, so an origin of `[8, 8, 8]` in the model file is `(0.5, 0.5,
    0.5)` here.

    *rescale* is the reference's own flag: it stretches the two axes across the
    rotation back out, which is what makes a 45-degree plane still reach the
    cell's corners.
    """
    a = math.radians(angle)
    c, s = math.cos(a), math.sin(a)
    k = 1.0 / math.cos(a) if rescale else 1.0
    ox, oy, oz = origin
    out = []
    for corners, uvs, slot, shade in quads:
        turned = []
        for x, y, z in corners:
            dx, dy, dz = x - ox, y - oy, z - oz
            if axis == 'x':
                dy, dz = (dy * c - dz * s) * k, (dy * s + dz * c) * k
            elif axis == 'y':
                dz, dx = (dz * c - dx * s) * k, (dz * s + dx * c) * k
            else:
                dx, dy = (dx * c - dy * s) * k, (dx * s + dy * c) * k
            turned.append((ox + dx, oy + dy, oz + dz))
        out.append(quad(turned, uvs, slot, shade))
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

# A quarter turn maps FRONT's brightness onto RIGHT's and back. Nothing else
# moves: FRONT and BACK share one number, RIGHT and LEFT the other, and TOP,
# BOTTOM and the reference's unshaded quads are already where they belong.
_SIDE_SWAP = {FACE_SHADE[FRONT]: FACE_SHADE[RIGHT],
              FACE_SHADE[RIGHT]: FACE_SHADE[FRONT]}


def _turn(quads, turns):
    """*turns* quarter turns about the cell's vertical axis, north -> west.

    The reference draws one model per oriented block and turns it in the block
    state; facings are separate shapes here, so this is that turn. Exact
    arithmetic rather than a 90-degree `spin` — the corners have to land back on
    the pixel grid or a door stops matching the wall beside it.
    """
    out = []
    for corners, uvs, slot, shade in quads:
        for _ in range(turns % 4):
            corners = tuple((z, y, 1.0 - x) for x, y, z in corners)
        out.append(quad(corners, uvs, slot,
                        _SIDE_SWAP.get(shade, shade) if turns % 2 else shade))
    return out


def _facings(quads, base='north'):
    """One model's four facings, spun off the one the reference actually draws."""
    b = FACING_NAMES.index(base)
    return {f: _turn(quads, (b - i) % 4) for i, f in enumerate(FACING_NAMES)}


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
    """`block/cactus` — each side is set a pixel *in along its own normal* and is
    otherwise a full-width cube face; the top and bottom are full cube faces.

    Which element carries which faces is the whole shape. The reference draws
    north/south off the box inset in z and east/west off the one inset in x, so
    every side still spans the cell corner to corner and only stands back from
    it. Read the other way round — inset in x but drawn as north/south — the
    quad count, the uvs and the winding are all still right, and the block comes
    out 14 pixels wide with a 1-pixel slit down each of its four corners.
    """
    return (box(0.0, 0.0, 1 * PX, 1.0, 1.0, 15 * PX, faces=(FRONT, BACK))
            + box(1 * PX, 0.0, 0.0, 15 * PX, 1.0, 1.0, faces=(RIGHT, LEFT))
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


def _px(*n):
    """The reference's pixel numbers, as block units. Reads like the model."""
    return tuple(v * PX for v in n)


def _torch_wall():
    """`block/template_torch_wall` — the torch's stick, leaned off the west wall.

    Not the floor torch's two full-cell planes: the wall model is one small box
    with the tile's 2x10 stick strip on all six of its faces, tilted 22.5
    degrees so it points up and away from the wall. The reference draws the west
    one and rotates it; `_facings` does that here.
    """
    strip = _px(7, 6, 9, 16)
    stick = box(*_px(-1, 3.5, 7, 1, 13.5, 9), slot=FRONT, shade=UNSHADED,
                uv={TOP: _px(7, 6, 9, 8), BOTTOM: _px(7, 13, 9, 15),
                    FRONT: strip, BACK: strip, RIGHT: strip, LEFT: strip})
    return spin(stick, _px(0, 3.5, 8), 'z', -22.5)


def _pot_body():
    """`block/flower_pot` — four thin walls around a plug of dirt.

    Two texture slots: TOP is the pot, BOTTOM the dirt inside it. FRONT is left
    for the plant, so a potted block's row reads (pot, pot, dirt, plant).
    """
    return (box(*_px(5, 0, 5, 6, 6, 11), slot=TOP,
                uv={BOTTOM: _px(5, 5, 6, 11), TOP: _px(5, 5, 6, 11),
                    BACK: _px(10, 10, 11, 16), FRONT: _px(5, 10, 6, 16),
                    LEFT: _px(5, 10, 11, 16), RIGHT: _px(5, 10, 11, 16)})
            + box(*_px(10, 0, 5, 11, 6, 11), slot=TOP,
                  uv={BOTTOM: _px(10, 5, 11, 11), TOP: _px(10, 5, 11, 11),
                      BACK: _px(5, 10, 6, 16), FRONT: _px(10, 10, 11, 16),
                      LEFT: _px(5, 10, 11, 16), RIGHT: _px(5, 10, 11, 16)})
            + box(*_px(6, 0, 5, 10, 6, 6), faces=(TOP, BOTTOM, FRONT, BACK),
                  slot=TOP,
                  uv={BOTTOM: _px(6, 10, 10, 11), TOP: _px(6, 5, 10, 6),
                      BACK: _px(6, 10, 10, 16), FRONT: _px(6, 10, 10, 16)})
            + box(*_px(6, 0, 10, 10, 6, 11), faces=(TOP, BOTTOM, FRONT, BACK),
                  slot=TOP,
                  uv={BOTTOM: _px(6, 5, 10, 6), TOP: _px(6, 10, 10, 11),
                      BACK: _px(6, 10, 10, 16), FRONT: _px(6, 10, 10, 16)})
            + box(*_px(6, 0, 6, 10, 4, 10), faces=(BOTTOM,), slot=TOP,
                  uv={BOTTOM: _px(6, 12, 10, 16)})
            + box(*_px(6, 0, 6, 10, 4, 10), faces=(TOP,), slot=BOTTOM,
                  uv={TOP: _px(6, 6, 10, 10)}))


def _pot_cross():
    """The plant in a pot — the `cross` model again, from y=4 up.

    The reference rescales it to the full diagonal exactly as a planted one, so
    a potted poppy is the same size as a poppy; only its feet are hidden.
    """
    y = 4 * PX
    a = ((0.0, y, 0.0), (1.0, y, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 0.0))
    b = ((1.0, y, 0.0), (0.0, y, 1.0), (0.0, 1.0, 1.0), (1.0, 1.0, 0.0))
    return double_sided(a, _PLANE_UV) + double_sided(b, _PLANE_UV)


def _lantern():
    """`block/template_lantern` — the cage, its cap, and two crossed hooks."""
    cage = box(*_px(5, 0, 5, 11, 7, 11), slot=FRONT,
               uv={BOTTOM: _px(0, 9, 6, 15), TOP: _px(0, 9, 6, 15),
                   BACK: _px(0, 2, 6, 9), FRONT: _px(0, 2, 6, 9),
                   LEFT: _px(0, 2, 6, 9), RIGHT: _px(0, 2, 6, 9)})
    cap = box(*_px(6, 7, 6, 10, 9, 10), faces=(TOP, FRONT, BACK, RIGHT, LEFT),
              slot=FRONT,
              uv={TOP: _px(1, 10, 5, 14), BACK: _px(1, 0, 5, 2),
                  FRONT: _px(1, 0, 5, 2), LEFT: _px(1, 0, 5, 2),
                  RIGHT: _px(1, 0, 5, 2)})
    hook_a = box(*_px(6.5, 9, 8, 9.5, 11, 8), faces=(FRONT, BACK), slot=FRONT,
                 shade=UNSHADED,
                 uv={BACK: _px(14, 1, 11, 3), FRONT: _px(11, 1, 14, 3)})
    hook_b = box(*_px(8, 9, 6.5, 8, 11, 9.5), faces=(RIGHT, LEFT), slot=FRONT,
                 shade=UNSHADED,
                 uv={LEFT: _px(14, 10, 11, 12), RIGHT: _px(11, 10, 14, 12)})
    centre = (0.5, 0.5, 0.5)
    return (cage + cap + spin(hook_a, centre, 'y', 45)
            + spin(hook_b, centre, 'y', 45))


def _chain():
    """`block/template_chain` — two narrow planes crossed at 45 degrees."""
    a = box(*_px(6.5, 0, 8, 9.5, 16, 8), faces=(FRONT, BACK), slot=FRONT,
            shade=UNSHADED,
            uv={BACK: _px(3, 0, 0, 16), FRONT: _px(0, 0, 3, 16)})
    b = box(*_px(8, 0, 6.5, 8, 16, 9.5), faces=(RIGHT, LEFT), slot=FRONT,
            shade=UNSHADED,
            uv={LEFT: _px(6, 0, 3, 16), RIGHT: _px(3, 0, 6, 16)})
    centre = (0.5, 0.5, 0.5)
    return spin(a, centre, 'y', 45) + spin(b, centre, 'y', 45)


def _end_rod():
    """`block/end_rod` — a 2x15 post standing on a 4x1 foot."""
    return (box(*_px(6, 0, 6, 10, 1, 10), slot=FRONT,
                uv={BOTTOM: _px(6, 6, 2, 2), TOP: _px(2, 2, 6, 6),
                    BACK: _px(2, 6, 6, 7), FRONT: _px(2, 6, 6, 7),
                    LEFT: _px(2, 6, 6, 7), RIGHT: _px(2, 6, 6, 7)})
            + box(*_px(7, 1, 7, 9, 16, 9),
                  faces=(TOP, FRONT, BACK, RIGHT, LEFT), slot=FRONT,
                  uv={TOP: _px(2, 0, 4, 2), BACK: _px(0, 0, 2, 15),
                      FRONT: _px(0, 0, 2, 15), LEFT: _px(0, 0, 2, 15),
                      RIGHT: _px(0, 0, 2, 15)}))


def _lightning_rod():
    """`block/template_lightning_rod` — a 2x12 post under a 4x4 head."""
    head = _px(0, 0, 4, 4)
    return (box(*_px(6, 12, 6, 10, 16, 10), slot=FRONT,
                uv={BACK: head, FRONT: head, LEFT: head, RIGHT: head,
                    BOTTOM: head, TOP: _px(4, 4, 0, 0)})
            + box(*_px(7, 0, 7, 9, 12, 9),
                  faces=(BOTTOM, FRONT, BACK, RIGHT, LEFT), slot=FRONT,
                  uv={BACK: _px(0, 4, 2, 16), FRONT: _px(0, 4, 2, 16),
                      LEFT: _px(0, 4, 2, 16), RIGHT: _px(0, 4, 2, 16),
                      BOTTOM: _px(0, 4, 2, 6)}))


def _blades(y, hinge_y, sign, slot):
    """Four flat blades hinged on the cell's edges and tilted 22.5 degrees.

    `block/coral_fan` and `block/spore_blossom` are the same construction at
    opposite ends of the cell — a fan tilting up off the floor, a blossom
    drooping under a ceiling, which is the whole of *sign*. Each blade is a full
    16x16 quad hinged on one edge of the cell, so its far half hangs a good way
    outside the block: that is the reference's own geometry and not a liberty
    taken here, and it is why a fan looks wider than the cell it is rooted in.
    """
    up, down = (0.0, 0.0, 1.0, 1.0), (0.0, 1.0, 1.0, 0.0)
    rev_up, rev_down = (1.0, 1.0, 0.0, 0.0), (1.0, 0.0, 0.0, 1.0)
    out = []
    for lo, hi, axis, angle, uv_up, uv_down in (
            (0.5, 1.5, 'z', -sign, up, down),
            (-0.5, 0.5, 'z', sign, up, down),
            (0.5, 1.5, 'x', sign, rev_up, rev_down),
            (-0.5, 0.5, 'x', -sign, up, down)):
        if axis == 'z':
            blade = box(lo, y, 0.0, hi, y, 1.0, faces=(TOP, BOTTOM), slot=slot,
                        shade=UNSHADED, uv={TOP: uv_up, BOTTOM: uv_down})
            hinge = (0.5, hinge_y, 0.0)
        else:
            blade = box(0.0, y, lo, 1.0, y, hi, faces=(TOP, BOTTOM), slot=slot,
                        shade=UNSHADED, uv={TOP: uv_up, BOTTOM: uv_down})
            hinge = (0.0, hinge_y, 0.5)
        out += spin(blade, hinge, axis, angle)
    return out


def _fan():
    """`block/coral_fan` — four blades tilting up off the floor."""
    return _blades(0.0, 0.0, -22.5, FRONT)


def _spore_blossom():
    """`block/spore_blossom` — the same four blades drooping off a ceiling,
    under a flat plate that hides where they meet."""
    y, inset = 15.9 * PX, 1 * PX
    plate = box(inset, y, inset, 1.0 - inset, y, 1.0 - inset,
                faces=(TOP, BOTTOM), slot=TOP, shade=UNSHADED,
                uv={TOP: (inset, inset, 1.0 - inset, 1.0 - inset),
                    BOTTOM: (inset, 1.0 - inset, 1.0 - inset, inset)})
    return plate + _blades(15.7 * PX, 1.0, 22.5, FRONT)


def _sea_pickle():
    """`block/sea_pickle` — one nub with a crossed tuft on top."""
    body = box(*_px(6, 0, 6, 10, 6, 10), slot=FRONT,
               uv={BOTTOM: _px(8, 1, 12, 5), TOP: _px(4, 1, 8, 5),
                   BACK: _px(4, 5, 8, 11), FRONT: _px(0, 5, 4, 11),
                   LEFT: _px(8, 5, 12, 11), RIGHT: _px(12, 5, 16, 11)})
    lid = box(*_px(6, 5.95, 6, 10, 5.95, 10), faces=(TOP,), slot=FRONT,
              uv={TOP: _px(8, 1, 12, 5)})
    a = box(*_px(7.5, 5.2, 8, 8.5, 8.7, 8), faces=(FRONT, BACK), slot=FRONT,
            shade=UNSHADED,
            uv={BACK: _px(1, 0, 3, 5), FRONT: _px(3, 0, 1, 5)})
    b = box(*_px(8, 5.2, 7.5, 8, 8.7, 8.5), faces=(RIGHT, LEFT), slot=FRONT,
            shade=UNSHADED,
            uv={LEFT: _px(13, 0, 15, 5), RIGHT: _px(15, 0, 13, 5)})
    centre = (0.5, 0.5, 0.5)
    return (body + lid + spin(a, centre, 'y', 45, rescale=True)
            + spin(b, centre, 'y', 45, rescale=True))


def _cake():
    """`block/cake` — a 14x8x14 block inset a pixel on every side."""
    return box(*_px(1, 0, 1, 15, 8, 15))


def _composter():
    """`block/composter` — four walls on a floor, open at the top."""
    out = box(*_px(0, 0, 0, 16, 2, 16), faces=(TOP, BOTTOM), slot=BOTTOM)
    for bounds, sides in ((_px(0, 0, 0, 2, 16, 16), (FRONT, BACK, RIGHT, LEFT)),
                          (_px(14, 0, 0, 16, 16, 16), (FRONT, BACK, RIGHT, LEFT)),
                          (_px(2, 0, 0, 14, 16, 2), (FRONT, BACK)),
                          (_px(2, 0, 14, 14, 16, 16), (FRONT, BACK))):
        out += box(*bounds, faces=(TOP,), slot=TOP)      # the rim
        out += box(*bounds, faces=sides, slot=FRONT)
    return out


def _anvil():
    """`block/template_anvil` — base, waist, neck and the block on top."""
    return (box(*_px(2, 0, 2, 14, 4, 14), slot=FRONT,
                uv={BOTTOM: _px(2, 2, 14, 14), TOP: _px(2, 2, 14, 14),
                    BACK: _px(2, 12, 14, 16), FRONT: _px(2, 12, 14, 16),
                    LEFT: _px(0, 2, 4, 14), RIGHT: _px(4, 2, 0, 14)})
            + box(*_px(4, 4, 3, 12, 5, 13),
                  faces=(TOP, FRONT, BACK, RIGHT, LEFT), slot=FRONT,
                  uv={TOP: _px(4, 3, 12, 13), BACK: _px(4, 11, 12, 12),
                      FRONT: _px(4, 11, 12, 12), LEFT: _px(4, 3, 5, 13),
                      RIGHT: _px(5, 3, 4, 13)})
            + box(*_px(6, 5, 4, 10, 10, 12), faces=(FRONT, BACK, RIGHT, LEFT),
                  slot=FRONT,
                  uv={BACK: _px(6, 6, 10, 11), FRONT: _px(6, 6, 10, 11),
                      LEFT: _px(5, 4, 10, 12), RIGHT: _px(10, 4, 5, 12)})
            + box(*_px(3, 10, 0, 13, 16, 16),
                  faces=(BOTTOM, FRONT, BACK, RIGHT, LEFT), slot=FRONT,
                  uv={BOTTOM: _px(3, 0, 13, 16), BACK: _px(3, 0, 13, 6),
                      FRONT: _px(3, 0, 13, 6), LEFT: _px(10, 0, 16, 16),
                      RIGHT: _px(16, 0, 10, 16)})
            + box(*_px(3, 10, 0, 13, 16, 16), faces=(TOP,), slot=TOP,
                  uv={TOP: _px(3, 0, 13, 16)}))


def _lectern():
    """`block/lectern` — a base, a post and the slanted reading top.

    Five texture names, so this is one of the two shapes whose block row spells
    all six faces out: TOP is the desk, BOTTOM the planks under it, FRONT the
    lectern's face, BACK its sides and RIGHT its base.
    """
    return (box(*_px(0, 0, 0, 16, 2, 16),
                faces=(TOP, FRONT, BACK, RIGHT, LEFT), slot=RIGHT,
                uv={BACK: _px(0, 14, 16, 16), RIGHT: _px(0, 6, 16, 8),
                    FRONT: _px(0, 6, 16, 8), LEFT: _px(0, 6, 16, 8),
                    TOP: _px(0, 0, 16, 16)})
            + box(*_px(0, 0, 0, 16, 2, 16), faces=(BOTTOM,), slot=BOTTOM,
                  uv={BOTTOM: _px(0, 0, 16, 16)})
            + box(*_px(4, 2, 4, 12, 15, 12), faces=(BACK, FRONT), slot=FRONT,
                  uv={BACK: _px(0, 0, 8, 13), FRONT: _px(8, 3, 16, 16)})
            + box(*_px(4, 2, 4, 12, 15, 12), faces=(RIGHT, LEFT), slot=BACK,
                  uv={RIGHT: _px(2, 16, 15, 8), LEFT: _px(2, 8, 15, 16)})
            + spin(box(*_px(0.0125, 12, 3, 15.9875, 16, 16),
                       faces=(BACK, RIGHT, FRONT, LEFT), slot=BACK,
                       uv={BACK: _px(0, 0, 16, 4), RIGHT: _px(0, 4, 13, 8),
                           FRONT: _px(0, 4, 16, 8), LEFT: _px(0, 4, 13, 8)})
                   + box(*_px(0.0125, 12, 3, 15.9875, 16, 16), faces=(TOP,),
                         slot=TOP, uv={TOP: _px(0, 1, 16, 14)})
                   + box(*_px(0.0125, 12, 3, 15.9875, 16, 16), faces=(BOTTOM,),
                         slot=BOTTOM, uv={BOTTOM: _px(0, 0, 16, 13)}),
                   (0.5, 0.5, 0.5), 'x', -22.5))


def _stonecutter():
    """`block/stonecutter` — a 9-high bench with the saw blade standing in it."""
    side = _px(0, 7, 16, 16)
    return (box(*_px(0, 0, 0, 16, 9, 16), slot=FRONT,
                uv={BOTTOM: _px(0, 0, 16, 16), TOP: _px(0, 0, 16, 16),
                    BACK: side, FRONT: side, LEFT: side, RIGHT: side})
            + box(*_px(1, 9, 8, 15, 16, 8), faces=(FRONT, BACK), slot=BACK,
                  uv={BACK: _px(1, 9, 15, 16), FRONT: _px(15, 9, 1, 16)}))


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
    'pot': _pot_body() + _pot_cross(),
    'pot_empty': _pot_body(),
    'lantern': _lantern(),
    'chain': _chain(),
    'end_rod': _end_rod(),
    'lightning_rod': _lightning_rod(),
    'fan': _fan(),
    'spore_blossom': _spore_blossom(),
    'sea_pickle': _sea_pickle(),
    'egg': box(*_px(5, 0, 4, 9, 7, 8), slot=FRONT,
               uv={BOTTOM: _px(0, 0, 4, 4), TOP: _px(0, 0, 4, 4),
                   BACK: _px(1, 4, 5, 11), FRONT: _px(1, 4, 5, 11),
                   LEFT: _px(1, 4, 5, 11), RIGHT: _px(1, 4, 5, 11)}),
    'cake': _cake(),
    'composter': _composter(),
    'anvil': _anvil(),
    # `block/enchanting_table` and `block/template_daylight_detector`: a full
    # cube cut down to 12 and 6 pixels. Both take their sides from the bottom of
    # the tile, which is what `box`'s automatic uv already does.
    'enchanting_table': box(*_px(0, 0, 0, 16, 12, 16)),
    'daylight_detector': box(*_px(0, 0, 0, 16, 6, 16)),
}
for _f in FACING_NAMES:
    SHAPES['door_' + _f] = box(*_DOOR_BOUNDS[_f], slot=FRONT)
    SHAPES['ladder_' + _f] = _ladder(_f)
# The oriented shapes: one model each, turned. The reference draws a wall torch
# on the west wall and everything else facing north.
for _f, _quads in _facings(_torch_wall(), base='west').items():
    SHAPES['torch_wall_' + _f] = _quads
for _f, _quads in _facings(_lectern()).items():
    SHAPES['lectern_' + _f] = _quads
for _f, _quads in _facings(_stonecutter()).items():
    SHAPES['stonecutter_' + _f] = _quads

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
                         'snow_layer', 'lily', 'pot', 'pot_empty', 'lantern',
                         'chain', 'end_rod', 'lightning_rod', 'fan',
                         'spore_blossom', 'sea_pickle', 'egg'}
                        | {'ladder_' + f for f in FACING_NAMES}
                        | {'torch_wall_' + f for f in FACING_NAMES})

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
# Not 0..1: the reference lets an element run from -16 to 32 pixels and several
# of these use it — a wall torch's heel is a pixel inside the wall it hangs on,
# a coral fan's blades are hinged on the cell's edges and reach half a block
# past them. Drawing outside the cell costs nothing here (the jitter on a plant
# already does it), so this is a typo check: a pixel number left un-scaled by PX
# still trips it.
assert POS.min() >= -0.6 and POS.max() <= 1.5, 'a shape is nowhere near its block'
assert UV.min() >= 0.0 and UV.max() <= 1.0, 'a shape samples outside its tile'
