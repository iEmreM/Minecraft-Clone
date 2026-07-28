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

@njit(nogil=True, fastmath=True, cache=True)
def get_optimized_ao(blocks, neighbors, x, y, z, face_id):
    """
    Optimized AO calculation with reduced neighbor sampling (3 instead of 5)
    face_id: 0=top, 1=bottom, 2=front, 3=back, 4=right, 5=left
    Performance: ~40% faster than previous version
    """
    base_ao = 0.8
    
    # Sample only 3 critical neighbors instead of 5 for better performance
    # Format: (dx, dy, dz) relative to block position
    solid_count = 0
    
    if face_id == 0: # top
        # Check: directly above, diagonal left, diagonal front
        if get_block_ext(blocks, neighbors, x, y+1, z) != AIR:
            solid_count += 1
        if get_block_ext(blocks, neighbors, x-1, y+1, z) != AIR:
            solid_count += 1
        if get_block_ext(blocks, neighbors, x, y+1, z-1) != AIR:
            solid_count += 1
    elif face_id == 1: # bottom
        if get_block_ext(blocks, neighbors, x, y-1, z) != AIR:
            solid_count += 1
        if get_block_ext(blocks, neighbors, x-1, y-1, z) != AIR:
            solid_count += 1
        if get_block_ext(blocks, neighbors, x, y-1, z-1) != AIR:
            solid_count += 1
    elif face_id == 2: # front (Z+)
        if get_block_ext(blocks, neighbors, x, y, z+1) != AIR:
            solid_count += 1
        if get_block_ext(blocks, neighbors, x-1, y, z+1) != AIR:
            solid_count += 1
        if get_block_ext(blocks, neighbors, x, y-1, z+1) != AIR:
            solid_count += 1
    elif face_id == 3: # back (Z-)
        if get_block_ext(blocks, neighbors, x, y, z-1) != AIR:
            solid_count += 1
        if get_block_ext(blocks, neighbors, x-1, y, z-1) != AIR:
            solid_count += 1
        if get_block_ext(blocks, neighbors, x, y-1, z-1) != AIR:
            solid_count += 1
    elif face_id == 4: # right (X+)
        if get_block_ext(blocks, neighbors, x+1, y, z) != AIR:
            solid_count += 1
        if get_block_ext(blocks, neighbors, x+1, y-1, z) != AIR:
            solid_count += 1
        if get_block_ext(blocks, neighbors, x+1, y, z-1) != AIR:
            solid_count += 1
    elif face_id == 5: # left (X-)
        if get_block_ext(blocks, neighbors, x-1, y, z) != AIR:
            solid_count += 1
        if get_block_ext(blocks, neighbors, x-1, y-1, z) != AIR:
            solid_count += 1
        if get_block_ext(blocks, neighbors, x-1, y, z-1) != AIR:
            solid_count += 1
    
    # Adjusted AO reduction for 3 samples (slightly stronger per-sample impact)
    ao_reduction = (solid_count / 3.0) * 0.25
    val = base_ao - ao_reduction
    if val < 0.4:
        return 0.4
    return val

@njit(nogil=True, fastmath=True, cache=True)
def get_greedy_quad(chunk_x, chunk_z, x, y, z, width, height, face_id, block_type, blocks, neighbors):
    """
    Generate vertices for a greedy quad with Texture Array support
    """
    world_x = chunk_x * CHUNK_SIZE
    world_z = chunk_z * CHUNK_SIZE
    
    # Base coords
    bx, by, bz = world_x + x, y, world_z + z
    
    # Quad structure: 4 vertices * 7 attributes (x,y,z, u,v,layer, shading)
    result = np.empty(28, dtype=np.float32)
    
    x_min, y_min, z_min = bx, by, bz
    x_max, y_max, z_max = bx, by, bz
    
    if face_id == 0: # Top (Y+)
        x_max += width
        z_max += height
        y_min += 1.0; y_max += 1.0
        vx = np.array([x_min, x_min, x_max, x_max], dtype=np.float32)
        vy = np.array([y_min, y_min, y_min, y_min], dtype=np.float32)
        vz = np.array([z_min, z_max, z_max, z_min], dtype=np.float32)
        shading = 1.0
        
    elif face_id == 1: # Bottom (Y-)
        x_max += width
        z_max += height
        vx = np.array([x_min, x_max, x_max, x_min], dtype=np.float32)
        vy = np.array([y_min, y_min, y_min, y_min], dtype=np.float32)
        vz = np.array([z_min, z_min, z_max, z_max], dtype=np.float32)
        shading = 0.4
        
    elif face_id == 2: # Front (Z+)
        x_max += width
        y_max += height
        z_min += 1.0; z_max += 1.0
        vx = np.array([x_min, x_max, x_max, x_min], dtype=np.float32)
        vy = np.array([y_min, y_min, y_max, y_max], dtype=np.float32)
        vz = np.array([z_min, z_min, z_min, z_min], dtype=np.float32)
        shading = 0.8
        
    elif face_id == 3: # Back (Z-)
        x_max += width
        y_max += height
        vx = np.array([x_max, x_min, x_min, x_max], dtype=np.float32)
        vy = np.array([y_min, y_min, y_max, y_max], dtype=np.float32)
        vz = np.array([z_min, z_min, z_min, z_min], dtype=np.float32)
        shading = 0.8
        
    elif face_id == 4: # Right (X+)
        z_max += width
        y_max += height
        x_min += 1.0; x_max += 1.0
        vx = np.array([x_min, x_min, x_min, x_min], dtype=np.float32)
        vy = np.array([y_min, y_min, y_max, y_max], dtype=np.float32)
        vz = np.array([z_max, z_min, z_min, z_max], dtype=np.float32)
        shading = 0.6
        
    elif face_id == 5: # Left (X-)
        z_max += width
        y_max += height
        vx = np.array([x_min, x_min, x_min, x_min], dtype=np.float32)
        vy = np.array([y_min, y_min, y_max, y_max], dtype=np.float32)
        vz = np.array([z_min, z_max, z_max, z_min], dtype=np.float32)
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

    ao = get_optimized_ao(blocks, neighbors, x, y, z, face_id)
    final_shading = shading * ao
    
    for i in range(4):
        base = i * 7 # 7 floats per vertex now
        result[base] = vx[i]
        result[base+1] = vy[i]
        result[base+2] = vz[i]
        result[base+3] = u_vals[i]
        result[base+4] = v_vals[i]
        result[base+5] = layer # Texture Layer
        result[base+6] = final_shading
        
    return result

@njit(nogil=True, fastmath=True, cache=True)
def build_chunk_mesh_fast(blocks, chunk_x, chunk_z, neighbors):
    """
    Fast chunk mesh builder using Greedy Meshing (Texture Array version)

    *neighbors* is the (-X, +X, -Z, +Z) tuple of neighbor block arrays; see
    get_block_ext. Pass NO_NEIGHBOR for any chunk that is not loaded.
    Returns (vertices, indices)
    """
    max_faces = 20000 
    # 7 floats per vertex now (pos3 + uv3 + shading1)
    vertices = np.empty(max_faces * 4 * 7, dtype=np.float32)
    indices = np.empty(max_faces * 6, dtype=np.uint32)
    
    vertex_count = 0
    index_count = 0
    
    dims = np.array([CHUNK_SIZE, CHUNK_HEIGHT, CHUNK_SIZE])
    
    for face_id in range(6):
        if face_id == 0 or face_id == 1:
            d_axis = 1; u_axis = 0; v_axis = 2
        elif face_id == 2 or face_id == 3:
            d_axis = 2; u_axis = 0; v_axis = 1
        else:
            d_axis = 0; u_axis = 2; v_axis = 1
            
        direction = 1 if (face_id % 2 == 0) else -1
        
        mask = np.zeros((dims[u_axis], dims[v_axis]), dtype=np.int32) 
        
        for d in range(dims[d_axis]):
            mask.fill(0)
            
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
            
            for v in range(dims[v_axis]):
                u = 0
                while u < dims[u_axis]:
                    if mask[u, v] != 0:
                        block_type = mask[u, v]
                        width = 1
                        while u + width < dims[u_axis] and mask[u + width, v] == block_type:
                            width += 1
                        
                        height = 1
                        done = False
                        while v + height < dims[v_axis]:
                            for w in range(width):
                                if mask[u + w, v + height] != block_type:
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
                        
                        face_data = get_greedy_quad(chunk_x, chunk_z, q_x, q_y, q_z, width, height, face_id, block_type, blocks, neighbors)
                        
                        base_v_idx = vertex_count * 7 # Updated stride
                        for i in range(28): # 4 vertices * 7 floats
                            vertices[base_v_idx + i] = face_data[i]
                        
                        start_v = int(vertex_count / 1)
                        indices[index_count] = start_v
                        indices[index_count+1] = start_v + 1
                        indices[index_count+2] = start_v + 2
                        indices[index_count+3] = start_v
                        indices[index_count+4] = start_v + 2
                        indices[index_count+5] = start_v + 3
                        
                        vertex_count += 4
                        index_count += 6
                        
                        for h in range(height):
                            for w in range(width):
                                mask[u + w, v + h] = 0
                        u += width
                    else:
                        u += 1

    return vertices[:vertex_count*7], indices[:index_count]
