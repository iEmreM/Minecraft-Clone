import numpy as np
from numba import njit
import math

from world.blocks import BLOCK_DTYPE
from world.fast_noise import fast_rand
from world.shapes import JITTER

# Chunk constants
CHUNK_SIZE = 16
CHUNK_HEIGHT = 256
AIR = 0
STONE = 3
WATER = 8

# How many solid blocks have to be stacked over a pocket of air before the far
# LOD treats it as out of sight and fills it back in. See seal_buried_air.
#
# 8 sits in a wide gap. Above it: the terrain generator only carves caves below
# `terrain_height - CAVE_ROOF` (9), so every cave is under a crust at least that
# thick and every cave is caught — `test_worldgen.py` asserts that relation from
# the other side, because a cave allowed nearer the surface would be filled in
# here and a cave mouth would close as the player walked away. Below it: the
# thickest thing that legitimately has air under it is a tree canopy, and a
# village roof is thinner still. Measured on 225 chunks,
# raising the number to 4 changes the quad count by 0.2% and lowering it to 16
# gives up half the saving — the terrain is not close to either edge.
SEAL_COVER = 8

# Stand-in for a neighbor chunk that is not loaded. All AIR, so the seam facing
# it is treated as exposed — the behaviour the mesher had before it knew about
# neighbors at all. Shared and never written to.
NO_NEIGHBOR = np.zeros((CHUNK_SIZE, CHUNK_HEIGHT, CHUNK_SIZE), dtype=BLOCK_DTYPE)

# Ceiling on quads per chunk. Generated terrain peaks around 3200, so this is
# mostly headroom for player-built geometry.
MAX_FACES = 20000

# The same, for the see-through pass — which is also where every non-cube block
# goes. Generated terrain does fill this one now: a meadow puts a plant on
# roughly a fifth of its columns and each is 4 quads, so a grassy chunk arrives
# with a few hundred already in it (225 chunks at seed 42: 90 on average, 308 on
# the worst, a jungle). The rest is headroom for a glass build on top of that.
MAX_FACES_ALPHA = 8000


def make_mesh_buffers():
    """Scratch space for build_chunk_mesh_fast: one set per meshing thread.

    Four buffers: vertices and indices for the opaque pass, then the same pair
    for the transparent one. The builder used to allocate these itself, so every
    chunk malloc'd and threw away 2.7 MB to fill about 7% of it. Owning them per
    caller keeps that off the hot path without sharing anything across threads.
    """
    return (np.empty(MAX_FACES * 4 * 7, dtype=np.float32),
            np.empty(MAX_FACES * 6, dtype=np.uint32),
            np.empty(MAX_FACES_ALPHA * 4 * 7, dtype=np.float32),
            np.empty(MAX_FACES_ALPHA * 6, dtype=np.uint32))

@njit(nogil=True, fastmath=True, cache=True)
def column_seal_limit(blocks, opaque, x, z, cover):
    """Highest y in this column that has *cover* solid blocks stacked above it,
    or -1 if the column never gets that deep.

    Because the count only ever grows going down, "buried under *cover* blocks"
    is a half-open range, and one number per column describes it. That is what
    lets the mesher seal a neighbor chunk it only ever reads one layer of.

    Only blocks you cannot see through count as cover, so a glass roof does not
    let the far LOD fill the cave under it back in — and **only terrain counts,
    not foliage**, which is what `opaque == 1` says (see blocks.OPAQUE, where 2
    is the leaf group). A canopy is not a cave roof: a mega spruce's crown is
    thirteen rows deep, so counting its leaves as cover put the limit *above the
    ground*, and everything under the tree — including the open air you can walk
    through between the skirt and the grass — came back as stone the moment the
    chunk dropped to the far LOD.
    """
    seen = 0
    for y in range(CHUNK_HEIGHT - 1, -1, -1):
        if opaque[blocks[x, y, z]] == 1:
            seen += 1
            if seen == cover:
                return y - 1
    return -1


@njit(nogil=True, fastmath=True, cache=True)
def buried_to_stone(block, y, limit):
    """Air at or below *limit* reads as stone: it is roofed over by enough rock
    that nothing outside the terrain can see it."""
    if y <= limit and (block == AIR or block == WATER):
        return STONE
    return block


@njit(nogil=True, fastmath=True, cache=True)
def seal_buried_air(blocks, opaque, cover):
    """Fill this chunk's buried air with stone, in place.

    Two thirds of a chunk's quads face a cave. None of them can be seen from
    outside the terrain, and filling the caves in deletes them without moving a
    single surface: the outline against the sky, every slope, every overhang lip
    and every tree is exactly where it was. That is the whole reason the far
    LOD does this instead of meshing a coarser grid — a coarser grid is cheaper
    still, but it changes the shape, and a changed shape is what reads as
    "distant terrain turned into big blocks".

    Stone is the right filler rather than a guess: `get_block_type` returns
    STONE for everything below `terrain_height - 1` and carves the caves out of
    it afterwards, so this restores the rock that the cave was cut from. A
    player's hollowed-out mountain would come back as stone too, but only while
    it is far enough away to be meshed at this level.

    Each column is read for its limit before it is written, and columns do not
    look at each other, so doing this in place is safe.
    """
    for x in range(CHUNK_SIZE):
        for z in range(CHUNK_SIZE):
            limit = column_seal_limit(blocks, opaque, x, z, cover)
            for y in range(limit + 1):
                block = blocks[x, y, z]
                if block == AIR or block == WATER:
                    blocks[x, y, z] = STONE


@njit(nogil=True, fastmath=True, cache=True)
def neighbor_seal_limits(neighbors, opaque, cover):
    """Seal limits for the one layer of each neighbor the mesher can reach.

    The sweep and the AO ring never sample more than a block past the seam, so
    a neighbor only ever contributes its touching layer — 16 columns instead of
    256. Sealing the neighbors matters: left raw, every cave that crosses a
    chunk boundary grows a wall of quads on the seam (measured: 783 quads a
    chunk becomes 1063).
    """
    limits = np.empty((4, CHUNK_SIZE), dtype=np.int32)
    for i in range(CHUNK_SIZE):
        limits[0, i] = column_seal_limit(neighbors[0], opaque, CHUNK_SIZE - 1, i, cover)
        limits[1, i] = column_seal_limit(neighbors[1], opaque, 0, i, cover)
        limits[2, i] = column_seal_limit(neighbors[2], opaque, i, CHUNK_SIZE - 1, cover)
        limits[3, i] = column_seal_limit(neighbors[3], opaque, i, 0, cover)
    return limits


@njit(nogil=True, fastmath=True, cache=True)
def get_block_ext(blocks, neighbors, nlimits, x, y, z):
    """Block at chunk-local (x, y, z), reaching into a neighbor when x or z is
    outside this chunk.

    *neighbors* is the 4-tuple (-X, +X, -Z, +Z) of neighbor block arrays, with
    NO_NEIGHBOR standing in for chunks that are not loaded. Without this the
    mesher saw every chunk in isolation: it treated the 4 seams as open air and
    emitted a full wall of invisible faces on each one, and AO along the seam
    was computed as if the neighbor were empty.

    *nlimits* is neighbor_seal_limits' table, applied on the way out so the
    neighbors look sealed without being copied. All -1 means no sealing, which
    is what the near LOD passes.

    Diagonal neighbors are not passed in, so a sample off the chunk on both x
    and z reads as AIR.
    ponytail: that leaves AO slightly wrong on the 4 corner columns of a chunk
    (1 column each, never noticed in play) — pass the diagonals in too if it
    ever shows up.
    """
    if y < 0 or y >= CHUNK_HEIGHT:
        return AIR
    if x < 0:
        if z < 0 or z >= CHUNK_SIZE:
            return AIR
        return buried_to_stone(neighbors[0][x + CHUNK_SIZE, y, z], y, nlimits[0, z])
    if x >= CHUNK_SIZE:
        if z < 0 or z >= CHUNK_SIZE:
            return AIR
        return buried_to_stone(neighbors[1][x - CHUNK_SIZE, y, z], y, nlimits[1, z])
    if z < 0:
        return buried_to_stone(neighbors[2][x, y, z + CHUNK_SIZE], y, nlimits[2, x])
    if z >= CHUNK_SIZE:
        return buried_to_stone(neighbors[3][x, y, z - CHUNK_SIZE], y, nlimits[3, x])
    return blocks[x, y, z]

# Corner shading, darkest to brightest. The open end stays at 0.8, so lit
# surfaces look exactly as they did — only the occluded end goes deeper. The
# steps widen as occlusion builds (0.09, 0.12, 0.14) instead of being evenly
# spaced, which is roughly how the visible sky falls away and makes a corner
# read as a contact shadow rather than a faint tint.
AO_LEVELS = (0.45, 0.59, 0.71, 0.80)

# Every corner at level 3 — the code the far LOD levels use instead of sampling
# AO, so an unoccluded face keeps exactly the brightness it has today.
AO_FULL = 0b11111111

@njit(nogil=True, fastmath=True, cache=True)
def get_face_ao(blocks, neighbors, nlimits, opaque, x, y, z, face_id):
    """Per-corner ambient occlusion for one block face, packed into an int.

    Returns the four corner levels (0 = wedged into a corner, 3 = open sky) at
    2 bits each, in the vertex order emit_greedy_quad writes them. Only opaque
    neighbors occlude — a pane of glass beside a floor must not darken it.

    Packing them into a single int is what lets the greedy pass compare two
    faces' shading with one ==, so blocks whose corners are lit differently are
    no longer merged into one flat quad. AO used to be sampled once per merged
    quad, which made a 16-block run take the shading of its first block.
    """
    # Face normal, then the two in-plane axes, in the order the quad's width and
    # height expand along.
    if face_id == 0:      # top (Y+):     width -> X, height -> Z
        nx, ny, nz = 0, 1, 0
        ux, uy, uz = 1, 0, 0
        vx, vy, vz = 0, 0, 1
    elif face_id == 1:    # bottom (Y-):  width -> X, height -> Z
        nx, ny, nz = 0, -1, 0
        ux, uy, uz = 1, 0, 0
        vx, vy, vz = 0, 0, 1
    elif face_id == 2:    # front (Z+):   width -> X, height -> Y
        nx, ny, nz = 0, 0, 1
        ux, uy, uz = 1, 0, 0
        vx, vy, vz = 0, 1, 0
    elif face_id == 3:    # back (Z-):    width -> X, height -> Y
        nx, ny, nz = 0, 0, -1
        ux, uy, uz = 1, 0, 0
        vx, vy, vz = 0, 1, 0
    elif face_id == 4:    # right (X+):   width -> Z, height -> Y
        nx, ny, nz = 1, 0, 0
        ux, uy, uz = 0, 0, 1
        vx, vy, vz = 0, 1, 0
    else:                 # left (X-):    width -> Z, height -> Y
        nx, ny, nz = -1, 0, 0
        ux, uy, uz = 0, 0, 1
        vx, vy, vz = 0, 1, 0

    # Which (u, v) corner each of the four vertices sits on, matching the vertex
    # order in emit_greedy_quad.
    if face_id == 0:
        su = (-1, -1, 1, 1)
        sv = (-1, 1, 1, -1)
    elif face_id == 3 or face_id == 4:
        su = (1, -1, -1, 1)
        sv = (-1, -1, 1, 1)
    else:                 # 1, 2, 5
        su = (-1, 1, 1, -1)
        sv = (-1, -1, 1, 1)

    # Everything is sampled in the layer of air the face looks into. The four
    # corners share their side samples, so the ring around the face is read once
    # (8 lookups) instead of 3 per corner.
    bx = x + nx
    by = y + ny
    bz = z + nz

    s_um = opaque[get_block_ext(blocks, neighbors, nlimits, bx - ux, by - uy, bz - uz)] != 0
    s_up = opaque[get_block_ext(blocks, neighbors, nlimits, bx + ux, by + uy, bz + uz)] != 0
    s_vm = opaque[get_block_ext(blocks, neighbors, nlimits, bx - vx, by - vy, bz - vz)] != 0
    s_vp = opaque[get_block_ext(blocks, neighbors, nlimits, bx + vx, by + vy, bz + vz)] != 0

    c_mm = opaque[get_block_ext(blocks, neighbors, nlimits, bx - ux - vx, by - uy - vy, bz - uz - vz)] != 0
    c_pm = opaque[get_block_ext(blocks, neighbors, nlimits, bx + ux - vx, by + uy - vy, bz + uz - vz)] != 0
    c_mp = opaque[get_block_ext(blocks, neighbors, nlimits, bx - ux + vx, by - uy + vy, bz - uz + vz)] != 0
    c_pp = opaque[get_block_ext(blocks, neighbors, nlimits, bx + ux + vx, by + uy + vy, bz + uz + vz)] != 0

    code = 0
    for i in range(4):
        au = su[i]
        av = sv[i]

        side_u = s_um if au < 0 else s_up
        side_v = s_vm if av < 0 else s_vp

        if side_u and side_v:
            level = 0     # both sides closed: the diagonal cannot let light in
        else:
            if au < 0:
                diag = c_mm if av < 0 else c_mp
            else:
                diag = c_pm if av < 0 else c_pp
            level = 3 - int(side_u) - int(side_v) - int(diag)

        code |= level << (2 * i)

    return code

@njit(nogil=True, fastmath=True, cache=True)
def emit_greedy_quad(vertices, offset, chunk_x, chunk_z, x, y, z, width, height, face_id, block_type, ao_code, face_layers):
    """
    Write one greedy quad's 4 vertices into *vertices* starting at *offset*.

    *ao_code* is the packed per-corner AO from get_face_ao; every block merged
    into this quad shares it.

    *face_layers* is blocks.FACE_LAYER — [block_type, face_id] -> atlas layer.
    It arrives as an argument rather than as a module global because
    @njit(cache=True) freezes globals into the cached artifact without
    invalidating it when they change, so a newly added block would silently
    keep the old block's texture.

    Writes in place and uses tuples for the corner coordinates: this used to
    allocate five small numpy arrays per quad, which cost about a third of the
    whole mesh build once per-corner AO pushed the quad count up.
    """
    world_x = chunk_x * CHUNK_SIZE
    world_z = chunk_z * CHUNK_SIZE

    # Base coords
    bx, by, bz = float(world_x + x), float(y), float(world_z + z)

    qw = float(width)
    qh = float(height)

    x_min, y_min, z_min = bx, by, bz
    x_max, y_max, z_max = bx, by, bz

    if face_id == 0: # Top (Y+)
        x_max += qw
        z_max += qh
        y_min += 1.0; y_max += 1.0
        vx = (x_min, x_min, x_max, x_max)
        vy = (y_min, y_min, y_min, y_min)
        vz = (z_min, z_max, z_max, z_min)
        shading = 1.0

    elif face_id == 1: # Bottom (Y-)
        x_max += qw
        z_max += qh
        vx = (x_min, x_max, x_max, x_min)
        vy = (y_min, y_min, y_min, y_min)
        vz = (z_min, z_min, z_max, z_max)
        shading = 0.4

    elif face_id == 2: # Front (Z+)
        x_max += qw
        y_max += qh
        z_min += 1.0; z_max += 1.0
        vx = (x_min, x_max, x_max, x_min)
        vy = (y_min, y_min, y_max, y_max)
        vz = (z_min, z_min, z_min, z_min)
        shading = 0.8

    elif face_id == 3: # Back (Z-)
        x_max += qw
        y_max += qh
        vx = (x_max, x_min, x_min, x_max)
        vy = (y_min, y_min, y_max, y_max)
        vz = (z_min, z_min, z_min, z_min)
        shading = 0.8

    elif face_id == 4: # Right (X+)
        z_max += qw
        y_max += qh
        x_min += 1.0; x_max += 1.0
        vx = (x_min, x_min, x_min, x_min)
        vy = (y_min, y_min, y_max, y_max)
        vz = (z_max, z_min, z_min, z_max)
        shading = 0.6

    elif face_id == 5: # Left (X-)
        z_max += qw
        y_max += qh
        vx = (x_min, x_min, x_min, x_min)
        vy = (y_min, y_min, y_max, y_max)
        vz = (z_min, z_max, z_max, z_min)
        shading = 0.6

    # Atlas layer for this block's face. One table lookup — the per-block
    # if-chain that used to live here had to grow a branch per block type, and
    # the same coordinates were repeated in engine/hud.py by hand.
    layer = float(face_layers[block_type, face_id])


    # UV Coordinates for Tiling
    # Simply 0 to the quad's world size, so one tile still covers one block
    u_min = 0.0
    v_min = 0.0
    u_max = qw
    v_max = qh
    
    if face_id == 0: # Top Face: xmin,zmin -> xmin,zmax -> xmax,zmax -> xmax,zmin
        u_vals = (u_min, u_min, u_max, u_max)
        v_vals = (v_min, v_max, v_max, v_min)
    elif face_id == 1: # Bottom Face: xmin,zmin -> xmax,zmin -> xmax,zmax -> xmin,zmax
        u_vals = (u_min, u_max, u_max, u_min)
        v_vals = (v_min, v_min, v_max, v_max)
    else: # Sides (Flipped V)
        u_vals = (u_min, u_max, u_max, u_min)
        v_vals = (v_max, v_max, v_min, v_min)

    for i in range(4):
        base = offset + i * 7 # 7 floats per vertex now
        vertices[base] = vx[i]
        vertices[base+1] = vy[i]
        vertices[base+2] = vz[i]
        vertices[base+3] = u_vals[i]
        vertices[base+4] = v_vals[i]
        vertices[base+5] = layer # Texture Layer
        # Face brightness times this corner's own occlusion.
        vertices[base+6] = shading * AO_LEVELS[(ao_code >> (2 * i)) & 3]

@njit(nogil=True, fastmath=True, cache=True)
def emit_shape_quads(vertices, offset, chunk_x, chunk_z, x, y, z, q0, q1,
                     block_type, face_layers, s_pos, s_uv, s_slot, s_shade,
                     jitter):
    """Copy one non-cube block's quads into *vertices*, starting at *offset*.

    The whole of the second draw path. A shape is a fixed list of quads in the
    block's own 0..1 cell (world/shapes.py), so there is nothing to merge, no
    neighbor to test and no AO to sample — this is a memcpy with the block's
    world position added and the atlas layer resolved per quad.

    *jitter* is the reference's `offset_type: XZ`, which every cross-shaped model
    carries: the plant is nudged up to a quarter block sideways from a hash of
    where it stands. Without it a meadow is a grid. It is the block's *world*
    position that is hashed, so the same tuft lands in the same place whichever
    chunk rebuild drew it.
    """
    wx = float(chunk_x * CHUNK_SIZE + x)
    wz = float(chunk_z * CHUNK_SIZE + z)

    dx = 0.0
    dz = 0.0
    if jitter != 0:
        dx = (fast_rand(wx, 0.0, wz) - 0.5) * 2.0 * JITTER
        dz = (fast_rand(wx, 1.0, wz) - 0.5) * 2.0 * JITTER

    at = offset
    for q in range(q0, q1):
        layer = float(face_layers[block_type, s_slot[q]])
        shade = s_shade[q]
        for i in range(4):
            vertices[at] = wx + s_pos[q, i, 0] + dx
            vertices[at + 1] = float(y) + s_pos[q, i, 1]
            vertices[at + 2] = wz + s_pos[q, i, 2] + dz
            vertices[at + 3] = s_uv[q, i, 0]
            vertices[at + 4] = s_uv[q, i, 1]
            vertices[at + 5] = layer
            vertices[at + 6] = shade
            at += 7


@njit(nogil=True, fastmath=True, cache=True)
def build_chunk_mesh_fast(blocks, chunk_x, chunk_z, neighbors, vertices, indices,
                          t_vertices, t_indices, face_layers, opaque, shape_table,
                          lod=0):
    """
    Fast chunk mesh builder using Greedy Meshing (Texture Array version)

    *neighbors* is the (-X, +X, -Z, +Z) tuple of neighbor block arrays; see
    get_block_ext. Pass NO_NEIGHBOR for any chunk that is not loaded.

    *face_layers* is blocks.FACE_LAYER, *opaque* is blocks.OPAQUE and
    *shape_table* is blocks.SHAPE_TABLE, all passed as arguments rather than
    read as globals — see emit_greedy_quad.

    **Two paths, not one.** `shape_table`'s first array says which: 0 is a cube
    and goes through the greedy sweep below exactly as before, -1 is not drawn
    at all (air, water), and anything positive is a torch, a flower or a door,
    drawn quad for quad by emit_shape_quads. That one lookup replaced the
    `!= AIR and != WATER` the mask loop used to open with, so the hot path pays
    nothing for the feature.

    The four scratch buffers come from make_mesh_buffers, one set per meshing
    thread. Returns right-sized copies of the filled part of each, so the
    scratch is free again on return: `(vertices, indices, t_vertices,
    t_indices)`, the second pair being the see-through blocks.

    **The two meshes are built in one sweep, not two.** A quad's pass is decided
    by `opaque[block_type]` at the moment it is emitted, and since a greedy run
    only ever merges one block type, a run is entirely one or entirely the
    other. That is why adding transparency costs the mesher nothing on terrain
    that has none — the second buffer simply stays empty.

    A face is emitted when the block on the other side cannot hide it:
    `opaque[neighbor] == 0`. A see-through block additionally drops the face it
    shares with its own kind, so a wall of glass is a pane, not a stack of
    boxes.

    *lod* drops work as the chunk gets further from the player. **Neither level
    moves a surface**: the outline against the sky, every slope and every tree
    is in exactly the same place at every level. All that goes is geometry
    nothing outside the terrain can see, and then shading detail that has shrunk
    below a pixel.

    | lod | what it drops                     | quads | what it costs to look at |
    | --: | :-------------------------------- | ----: | :----------------------- |
    |   0 | nothing                           |  2251 | —                        |
    |   1 | caves and other buried air        |   783 | nothing visible          |
    |   2 | + per-corner AO, one tone a face  |   547 | contact shadows          |

    (Quads per chunk, measured over 225 chunks of seed 42.)

    Level 1 is the big one, and it is free: two thirds of a chunk's quads face a
    cave. Level 2 helps for a second reason beyond skipping the AO samples —
    a greedy run stops wherever the AO code changes, so one tone per face lets
    a whole hillside collapse into a few quads.

    Meshing a downsampled grid was tried here and removed. It is cheaper again
    (346 quads), but a 2-block cell is 6 px of error at 190 blocks, and the
    distance reads as "the terrain turned into bigger blocks". Sealing gets
    within 60% of the same saving for no shape change at all, so there is no
    reason to spend the silhouette.
    """
    seal = lod >= 1
    use_ao = lod < 2

    if seal:
        # A copy, because the caller's array is the live chunk — and a snapshot
        # anyway, which is what this thread wants while the main thread may be
        # swapping a player edit in underneath it.
        blocks = blocks.copy()
        seal_buried_air(blocks, opaque, SEAL_COVER)
        nlimits = neighbor_seal_limits(neighbors, opaque, SEAL_COVER)
    else:
        nlimits = np.full((4, CHUNK_SIZE), -1, dtype=np.int32)

    max_vertices = MAX_FACES * 4
    max_t_vertices = MAX_FACES_ALPHA * 4

    vertex_count = 0
    index_count = 0
    t_vertex_count = 0
    t_index_count = 0
    # Numba compiles with boundscheck off, so running past the scratch buffer
    # would silently write into whatever follows it — and the buffer is reused
    # by the next chunk. Stop instead; the chunk loses its furthest faces, which
    # only a deliberately built checkerboard could ever reach.
    overflow = False

    # Only scan up to the highest block plus one layer of air for its top face.
    # Chunks are 256 tall but terrain tops out around 30-80, so the untrimmed
    # sweep spent most of its time on empty sky.
    max_y = 0
    for lx in range(CHUNK_SIZE):
        for lz in range(CHUNK_SIZE):
            for ly in range(CHUNK_HEIGHT - 1, max_y, -1):
                if blocks[lx, ly, lz] != AIR:
                    max_y = ly
                    break

    scan_height = max_y + 2
    if scan_height > CHUNK_HEIGHT:
        scan_height = CHUNK_HEIGHT

    # --- the blocks that are not cubes -------------------------------------
    # One sweep, ahead of the greedy one, over the same bounded height. Every
    # shape carries alpha, so every quad goes into the see-through buffer and is
    # drawn by the blended pass — the opaque terrain pass never sees them and
    # keeps its early-Z.
    #
    # A full shape or none of it: dropping half a torch would leave a floating
    # flame. The buffer filling up skips that block and lets the sweep carry on,
    # for the same reason the greedy pass does — see MAX_FACES_ALPHA.
    #
    # Level 2 drops them, and it is the one thing any level removes rather than
    # hides. It starts at 16 chunks; at 1200x800 and a 65 degree FOV a block
    # covers 628/d pixels, so a tuft of grass out there is 2.5 px of a fogged
    # silhouette, and it is a quarter of what the far ring draws (152 quads a
    # chunk against 624). Ground cover is also the only thing in the world that
    # is *only* detail — dropping a distant tree would leave a hole in a wood.
    shape_of, s_start, s_pos, s_uv, s_slot, s_shade, s_jitter = shape_table
    for lx in range(CHUNK_SIZE if lod < 2 else 0):
        for ly in range(scan_height):
            for lz in range(CHUNK_SIZE):
                block_type = blocks[lx, ly, lz]
                kind = shape_of[block_type]
                if kind <= 0:
                    continue
                q0 = s_start[kind]
                n_quads = s_start[kind + 1] - q0
                if t_vertex_count + 4 * n_quads > max_t_vertices:
                    continue
                emit_shape_quads(t_vertices, t_vertex_count * 7, chunk_x, chunk_z,
                                 lx, ly, lz, q0, q0 + n_quads, block_type,
                                 face_layers, s_pos, s_uv, s_slot, s_shade,
                                 s_jitter[kind])
                for j in range(n_quads):
                    corner = t_vertex_count + j * 4
                    t_indices[t_index_count] = corner
                    t_indices[t_index_count + 1] = corner + 1
                    t_indices[t_index_count + 2] = corner + 2
                    t_indices[t_index_count + 3] = corner
                    t_indices[t_index_count + 4] = corner + 2
                    t_indices[t_index_count + 5] = corner + 3
                    t_index_count += 6
                t_vertex_count += 4 * n_quads

    dims = np.array([CHUNK_SIZE, scan_height, CHUNK_SIZE])

    for face_id in range(6):
        if overflow:
            break

        if face_id == 0 or face_id == 1:
            d_axis = 1; u_axis = 0; v_axis = 2
        elif face_id == 2 or face_id == 3:
            d_axis = 2; u_axis = 0; v_axis = 1
        else:
            d_axis = 0; u_axis = 2; v_axis = 1
            
        direction = 1 if (face_id % 2 == 0) else -1
        
        mask = np.zeros((dims[u_axis], dims[v_axis]), dtype=np.int32)
        # Packed per-corner AO, kept beside the block type so a run only merges
        # while both match.
        mask_ao = np.zeros((dims[u_axis], dims[v_axis]), dtype=np.int32)

        for d in range(dims[d_axis]):
            if overflow:
                break

            mask.fill(0)
            mask_ao.fill(0)
            
            for u in range(dims[u_axis]):
                for v in range(dims[v_axis]):
                    if d_axis == 0:
                        x, y, z = d, v, u
                    elif d_axis == 1:
                        x, y, z = u, d, v
                    else:
                        x, y, z = u, v, d
                    
                    block_type = blocks[x, y, z]

                    # Cubes only: air and water are not drawn here at all, and
                    # the shapes were drawn above.
                    if shape_of[block_type] == 0:
                        nx, ny, nz = x, y, z
                        if d_axis == 0: nx += direction
                        elif d_axis == 1: ny += direction
                        elif d_axis == 2: nz += direction

                        # Looks into the neighbor chunk on the seams, so a face
                        # covered by the chunk next door is no longer emitted.
                        neighbor_type = get_block_ext(blocks, neighbors, nlimits, nx, ny, nz)
                        # Exposed unless the neighbor can hide it — and a
                        # see-through block also hides its own kind, so the
                        # inside of a glass wall is not drawn.
                        if opaque[neighbor_type] == 0 and neighbor_type != block_type:
                            mask[u, v] = block_type
                            # Flat shading past the near LOD: sampling AO costs
                            # 8 lookups a face, and — the real win — a run only
                            # merges while the AO code matches, so a uniform
                            # code lets the greedy pass swallow whole hillsides.
                            if use_ao:
                                mask_ao[u, v] = get_face_ao(blocks, neighbors, nlimits, opaque, x, y, z, face_id)
                            else:
                                mask_ao[u, v] = AO_FULL
            
            for v in range(dims[v_axis]):
                if overflow:
                    break

                u = 0
                while u < dims[u_axis]:
                    if mask[u, v] != 0:
                        see_through = opaque[mask[u, v]] == 0
                        if not see_through and vertex_count + 4 > max_vertices:
                            overflow = True
                            break
                        # A full see-through buffer drops its own quads and lets
                        # the sweep carry on. Stopping the whole chunk instead
                        # would trade a wall of glass for a hole in the terrain,
                        # and the glass buffer is the small one.
                        drop = see_through and t_vertex_count + 4 > max_t_vertices

                        block_type = mask[u, v]
                        ao_code = mask_ao[u, v]
                        width = 1
                        while (u + width < dims[u_axis] and mask[u + width, v] == block_type
                               and mask_ao[u + width, v] == ao_code):
                            width += 1

                        height = 1
                        done = False
                        while v + height < dims[v_axis]:
                            for w in range(width):
                                if (mask[u + w, v + height] != block_type
                                        or mask_ao[u + w, v + height] != ao_code):
                                    done = True
                                    break
                            if done:
                                break
                            height += 1
                        
                        if d_axis == 0:
                            q_x, q_y, q_z = d, v, u
                        elif d_axis == 1:
                            q_x, q_y, q_z = u, d, v
                        else:
                            q_x, q_y, q_z = u, v, d
                        
                        if not drop:
                            # Same geometry either way; only which buffer it
                            # lands in differs, and that is what keeps the
                            # opaque pass free of anything to blend or discard.
                            out_v = t_vertices if see_through else vertices
                            out_i = t_indices if see_through else indices
                            start_v = t_vertex_count if see_through else vertex_count
                            at_i = t_index_count if see_through else index_count

                            emit_greedy_quad(out_v, start_v * 7, chunk_x, chunk_z,
                                             q_x, q_y, q_z, width, height, face_id,
                                             block_type, ao_code, face_layers)

                            # Split the quad along its brighter diagonal. Cutting
                            # across the dark one smears a single dark corner over
                            # half the face.
                            ao_0 = ao_code & 3
                            ao_1 = (ao_code >> 2) & 3
                            ao_2 = (ao_code >> 4) & 3
                            ao_3 = (ao_code >> 6) & 3

                            if ao_0 + ao_2 >= ao_1 + ao_3:
                                out_i[at_i] = start_v
                                out_i[at_i+1] = start_v + 1
                                out_i[at_i+2] = start_v + 2
                                out_i[at_i+3] = start_v
                                out_i[at_i+4] = start_v + 2
                                out_i[at_i+5] = start_v + 3
                            else:
                                out_i[at_i] = start_v + 1
                                out_i[at_i+1] = start_v + 2
                                out_i[at_i+2] = start_v + 3
                                out_i[at_i+3] = start_v + 1
                                out_i[at_i+4] = start_v + 3
                                out_i[at_i+5] = start_v

                            if see_through:
                                t_vertex_count += 4
                                t_index_count += 6
                            else:
                                vertex_count += 4
                                index_count += 6

                        for h in range(height):
                            for w in range(width):
                                mask[u + w, v + h] = 0
                        u += width
                    else:
                        u += 1

    # Copies, not views: the scratch buffer is about to be reused, and a view
    # would also pin all 2.7 MB of it for as long as the mesh sits in a queue.
    return (vertices[:vertex_count*7].copy(), indices[:index_count].copy(),
            t_vertices[:t_vertex_count*7].copy(), t_indices[:t_index_count].copy())
