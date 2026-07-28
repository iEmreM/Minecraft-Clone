import numpy as np
from numba import njit
import math

# Chunk constants
CHUNK_SIZE = 16
CHUNK_HEIGHT = 256
AIR = 0
WATER = 8

# Stand-in for a neighbor chunk that is not loaded. All AIR, so the seam facing
# it is treated as exposed — the behaviour the mesher had before it knew about
# neighbors at all. Shared and never written to.
NO_NEIGHBOR = np.zeros((CHUNK_SIZE, CHUNK_HEIGHT, CHUNK_SIZE), dtype=np.uint8)

# Ceiling on quads per chunk. Generated terrain peaks around 3200, so this is
# mostly headroom for player-built geometry.
MAX_FACES = 20000


def make_mesh_buffers():
    """Scratch space for build_chunk_mesh_fast: one set per meshing thread.

    The builder used to allocate these itself, so every chunk malloc'd and threw
    away 2.7 MB to fill about 7% of it. Owning them per caller keeps that off
    the hot path without sharing anything across threads.
    """
    return (np.empty(MAX_FACES * 4 * 7, dtype=np.float32),
            np.empty(MAX_FACES * 6, dtype=np.uint32))

@njit(nogil=True, fastmath=True, cache=True)
def get_block_ext(blocks, neighbors, x, y, z):
    """Block at chunk-local (x, y, z), reaching into a neighbor when x or z is
    outside this chunk.

    *neighbors* is the 4-tuple (-X, +X, -Z, +Z) of neighbor block arrays, with
    NO_NEIGHBOR standing in for chunks that are not loaded. Without this the
    mesher saw every chunk in isolation: it treated the 4 seams as open air and
    emitted a full wall of invisible faces on each one, and AO along the seam
    was computed as if the neighbor were empty.

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
        return neighbors[0][x + CHUNK_SIZE, y, z]
    if x >= CHUNK_SIZE:
        if z < 0 or z >= CHUNK_SIZE:
            return AIR
        return neighbors[1][x - CHUNK_SIZE, y, z]
    if z < 0:
        return neighbors[2][x, y, z + CHUNK_SIZE]
    if z >= CHUNK_SIZE:
        return neighbors[3][x, y, z - CHUNK_SIZE]
    return blocks[x, y, z]

# Corner shading, darkest to brightest. The open end stays at 0.8, so lit
# surfaces look exactly as they did — only the occluded end goes deeper. The
# steps widen as occlusion builds (0.09, 0.12, 0.14) instead of being evenly
# spaced, which is roughly how the visible sky falls away and makes a corner
# read as a contact shadow rather than a faint tint.
AO_LEVELS = (0.45, 0.59, 0.71, 0.80)

@njit(nogil=True, fastmath=True, cache=True)
def get_face_ao(blocks, neighbors, x, y, z, face_id):
    """Per-corner ambient occlusion for one block face, packed into an int.

    Returns the four corner levels (0 = wedged into a corner, 3 = open sky) at
    2 bits each, in the vertex order emit_greedy_quad writes them.

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

    s_um = get_block_ext(blocks, neighbors, bx - ux, by - uy, bz - uz) != AIR
    s_up = get_block_ext(blocks, neighbors, bx + ux, by + uy, bz + uz) != AIR
    s_vm = get_block_ext(blocks, neighbors, bx - vx, by - vy, bz - vz) != AIR
    s_vp = get_block_ext(blocks, neighbors, bx + vx, by + vy, bz + vz) != AIR

    c_mm = get_block_ext(blocks, neighbors, bx - ux - vx, by - uy - vy, bz - uz - vz) != AIR
    c_pm = get_block_ext(blocks, neighbors, bx + ux - vx, by + uy - vy, bz + uz - vz) != AIR
    c_mp = get_block_ext(blocks, neighbors, bx - ux + vx, by - uy + vy, bz - uz + vz) != AIR
    c_pp = get_block_ext(blocks, neighbors, bx + ux + vx, by + uy + vy, bz + uz + vz) != AIR

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
def emit_greedy_quad(vertices, offset, chunk_x, chunk_z, x, y, z, width, height, face_id, block_type, ao_code):
    """
    Write one greedy quad's 4 vertices into *vertices* starting at *offset*.

    *ao_code* is the packed per-corner AO from get_face_ao; every block merged
    into this quad shares it.

    Writes in place and uses tuples for the corner coordinates: this used to
    allocate five small numpy arrays per quad, which cost about a third of the
    whole mesh build once per-corner AO pushed the quad count up.
    """
    world_x = chunk_x * CHUNK_SIZE
    world_z = chunk_z * CHUNK_SIZE

    # Base coords
    bx, by, bz = float(world_x + x), float(y), float(world_z + z)

    x_min, y_min, z_min = bx, by, bz
    x_max, y_max, z_max = bx, by, bz
    
    if face_id == 0: # Top (Y+)
        x_max += width
        z_max += height
        y_min += 1.0; y_max += 1.0
        vx = (x_min, x_min, x_max, x_max)
        vy = (y_min, y_min, y_min, y_min)
        vz = (z_min, z_max, z_max, z_min)
        shading = 1.0
        
    elif face_id == 1: # Bottom (Y-)
        x_max += width
        z_max += height
        vx = (x_min, x_max, x_max, x_min)
        vy = (y_min, y_min, y_min, y_min)
        vz = (z_min, z_min, z_max, z_max)
        shading = 0.4
        
    elif face_id == 2: # Front (Z+)
        x_max += width
        y_max += height
        z_min += 1.0; z_max += 1.0
        vx = (x_min, x_max, x_max, x_min)
        vy = (y_min, y_min, y_max, y_max)
        vz = (z_min, z_min, z_min, z_min)
        shading = 0.8
        
    elif face_id == 3: # Back (Z-)
        x_max += width
        y_max += height
        vx = (x_max, x_min, x_min, x_max)
        vy = (y_min, y_min, y_max, y_max)
        vz = (z_min, z_min, z_min, z_min)
        shading = 0.8
        
    elif face_id == 4: # Right (X+)
        z_max += width
        y_max += height
        x_min += 1.0; x_max += 1.0
        vx = (x_min, x_min, x_min, x_min)
        vy = (y_min, y_min, y_max, y_max)
        vz = (z_max, z_min, z_min, z_max)
        shading = 0.6
        
    elif face_id == 5: # Left (X-)
        z_max += width
        y_max += height
        vx = (x_min, x_min, x_min, x_min)
        vy = (y_min, y_min, y_max, y_max)
        vz = (z_min, z_max, z_max, z_min)
        shading = 0.6

    # Texture Layer Logic
    # 4x4 Atlas
    tex_x = 0; tex_y = 0
    
    # Block IDs
    GRASS = 1
    DIRT = 2
    STONE = 3
    SAND = 4
    SNOW = 5
    LEAVES = 6
    WOOD = 7
    WATER = 8
    STONE_BRICK = 9
    BRICK = 10
    
    if block_type == GRASS:
        if face_id == 0: tex_x, tex_y = 1, 3
        elif face_id == 1: tex_x, tex_y = 0, 2
        else: tex_x, tex_y = 0, 3
    elif block_type == DIRT: tex_x, tex_y = 0, 2
    elif block_type == STONE: tex_x, tex_y = 0, 1
    elif block_type == SAND: tex_x, tex_y = 1, 2
    elif block_type == SNOW: tex_x, tex_y = 3, 0
    elif block_type == LEAVES: tex_x, tex_y = 1, 0
    elif block_type == WOOD:
        if face_id == 0 or face_id == 1: tex_x, tex_y = 0, 0
        else: tex_x, tex_y = 2, 1
    elif block_type == WATER: tex_x, tex_y = 3, 0
    elif block_type == STONE_BRICK: tex_x, tex_y = 2, 2
    elif block_type == BRICK: tex_x, tex_y = 2, 3

    # Calculate layer index (row-major 4x4)
    # Assumes create_texture_array iterates y then x
    layer = float(tex_x + tex_y * 4)
    
    # UV Coordinates for Tiling
    # Simply 0 to width/height
    u_min = 0.0
    v_min = 0.0
    u_max = float(width)
    v_max = float(height)
    
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
def build_chunk_mesh_fast(blocks, chunk_x, chunk_z, neighbors, vertices, indices):
    """
    Fast chunk mesh builder using Greedy Meshing (Texture Array version)

    *neighbors* is the (-X, +X, -Z, +Z) tuple of neighbor block arrays; see
    get_block_ext. Pass NO_NEIGHBOR for any chunk that is not loaded.

    *vertices* and *indices* are reusable scratch space from make_mesh_buffers;
    the caller keeps one set per meshing thread. Returns fresh right-sized
    copies of the part that was filled, so the scratch is free again on return.
    """
    max_faces = MAX_FACES

    vertex_count = 0
    index_count = 0

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

    dims = np.array([CHUNK_SIZE, scan_height, CHUNK_SIZE])
    
    for face_id in range(6):
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
                    
                    if block_type != AIR and block_type != WATER:
                        nx, ny, nz = x, y, z
                        if d_axis == 0: nx += direction
                        elif d_axis == 1: ny += direction
                        elif d_axis == 2: nz += direction

                        # Looks into the neighbor chunk on the seams, so a face
                        # covered by the chunk next door is no longer emitted.
                        neighbor_type = get_block_ext(blocks, neighbors, nx, ny, nz)
                        if neighbor_type == AIR or neighbor_type == WATER:
                            mask[u, v] = block_type
                            mask_ao[u, v] = get_face_ao(blocks, neighbors, x, y, z, face_id)
            
            for v in range(dims[v_axis]):
                u = 0
                while u < dims[u_axis]:
                    if mask[u, v] != 0:
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
                        
                        emit_greedy_quad(vertices, vertex_count * 7, chunk_x, chunk_z,
                                         q_x, q_y, q_z, width, height, face_id,
                                         block_type, ao_code)

                        # Split the quad along its brighter diagonal. Cutting
                        # across the dark one smears a single dark corner over
                        # half the face.
                        ao_0 = ao_code & 3
                        ao_1 = (ao_code >> 2) & 3
                        ao_2 = (ao_code >> 4) & 3
                        ao_3 = (ao_code >> 6) & 3

                        start_v = vertex_count
                        if ao_0 + ao_2 >= ao_1 + ao_3:
                            indices[index_count] = start_v
                            indices[index_count+1] = start_v + 1
                            indices[index_count+2] = start_v + 2
                            indices[index_count+3] = start_v
                            indices[index_count+4] = start_v + 2
                            indices[index_count+5] = start_v + 3
                        else:
                            indices[index_count] = start_v + 1
                            indices[index_count+1] = start_v + 2
                            indices[index_count+2] = start_v + 3
                            indices[index_count+3] = start_v + 1
                            indices[index_count+4] = start_v + 3
                            indices[index_count+5] = start_v
                        
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
    return vertices[:vertex_count*7].copy(), indices[:index_count].copy()
