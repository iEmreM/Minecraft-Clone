"""
Ambient Occlusion system for realistic lighting
Based on ornek2's implementation - checks surrounding blocks 
to determine how much each vertex should be darkened
"""

def get_ao_value(blocks, x, y, z, face, vertex_index, chunk_size=16):
    """
    Calculate ambient occlusion value for a specific vertex of a face
    Returns a value between 0.0 (fully dark) and 1.0 (fully lit)
    """
    
    # Get the 8 surrounding positions for this face and vertex
    neighbors = get_ao_neighbors(x, y, z, face, vertex_index)
    
    # Count how many surrounding blocks are solid
    solid_count = 0
    for nx, ny, nz in neighbors:
        if is_block_solid(blocks, nx, ny, nz, chunk_size):
            solid_count += 1
    
    # Convert to AO value (more solid neighbors = darker)
    # 0 solid neighbors = 1.0 (bright)
    # 8 solid neighbors = 0.3 (dark but not completely black)
    ao_value = 1.0 - (solid_count / 8.0) * 0.7
    
    return ao_value

def get_ao_neighbors(x, y, z, face, vertex_index):
    """
    Get the 8 neighbor positions that affect AO for a specific vertex
    Based on the face orientation and vertex position
    """
    
    if face == 'top':  # Y+ face
        if vertex_index == 0:  # Bottom-left vertex
            return [(x-1, y, z-1), (x-1, y, z), (x-1, y, z+1),
                   (x, y, z-1), (x, y, z+1),
                   (x+1, y, z-1), (x+1, y, z), (x+1, y, z+1)]
        elif vertex_index == 1:  # Bottom-right vertex  
            return [(x-1, y, z-1), (x-1, y, z), (x-1, y, z+1),
                   (x, y, z-1), (x, y, z+1),
                   (x+1, y, z-1), (x+1, y, z), (x+1, y, z+1)]
        elif vertex_index == 2:  # Top-right vertex
            return [(x-1, y, z-1), (x-1, y, z), (x-1, y, z+1),
                   (x, y, z-1), (x, y, z+1),
                   (x+1, y, z-1), (x+1, y, z), (x+1, y, z+1)]
        else:  # Top-left vertex (index 3)
            return [(x-1, y, z-1), (x-1, y, z), (x-1, y, z+1),
                   (x, y, z-1), (x, y, z+1),
                   (x+1, y, z-1), (x+1, y, z), (x+1, y, z+1)]
    
    elif face == 'bottom':  # Y- face
        # Similar pattern but for bottom face
        return [(x-1, y, z-1), (x-1, y, z), (x-1, y, z+1),
               (x, y, z-1), (x, y, z+1),
               (x+1, y, z-1), (x+1, y, z), (x+1, y, z+1)]
    
    elif face == 'front':  # Z+ face  
        if vertex_index == 0:  # Bottom-left
            return [(x-1, y-1, z), (x-1, y, z), (x-1, y+1, z),
                   (x, y-1, z), (x, y+1, z),
                   (x+1, y-1, z), (x+1, y, z), (x+1, y+1, z)]
        elif vertex_index == 1:  # Bottom-right
            return [(x-1, y-1, z), (x-1, y, z), (x-1, y+1, z),
                   (x, y-1, z), (x, y+1, z),
                   (x+1, y-1, z), (x+1, y, z), (x+1, y+1, z)]
        elif vertex_index == 2:  # Top-right
            return [(x-1, y-1, z), (x-1, y, z), (x-1, y+1, z),
                   (x, y-1, z), (x, y+1, z),
                   (x+1, y-1, z), (x+1, y, z), (x+1, y+1, z)]
        else:  # Top-left
            return [(x-1, y-1, z), (x-1, y, z), (x-1, y+1, z),
                   (x, y-1, z), (x, y+1, z),
                   (x+1, y-1, z), (x+1, y, z), (x+1, y+1, z)]
    
    elif face == 'back':  # Z- face
        return [(x-1, y-1, z), (x-1, y, z), (x-1, y+1, z),
               (x, y-1, z), (x, y+1, z),
               (x+1, y-1, z), (x+1, y, z), (x+1, y+1, z)]
    
    elif face == 'right':  # X+ face
        return [(x, y-1, z-1), (x, y-1, z), (x, y-1, z+1),
               (x, y, z-1), (x, y, z+1),
               (x, y+1, z-1), (x, y+1, z), (x, y+1, z+1)]
    
    elif face == 'left':  # X- face
        return [(x, y-1, z-1), (x, y-1, z), (x, y-1, z+1),
               (x, y, z-1), (x, y, z+1),
               (x, y+1, z-1), (x, y+1, z), (x, y+1, z+1)]
    
    # Default fallback
    return []

def is_block_solid(blocks, x, y, z, chunk_size):
    """
    Check if a block is solid (not air) for AO calculation
    """
    # Check bounds
    if x < 0 or x >= chunk_size or y < 0 or y >= 256 or z < 0 or z >= chunk_size:
        return False  # Out of bounds blocks don't contribute to AO
    
    # Check if block is solid (not air)
    return blocks[x, y, z] != 0  # 0 = AIR

def calculate_face_ao(blocks, x, y, z, face, chunk_size=16):
    """
    Calculate AO values for all 4 vertices of a face
    Returns tuple of (ao0, ao1, ao2, ao3) for the four corners
    """
    ao_values = []
    
    for vertex_index in range(4):
        ao_value = get_ao_value(blocks, x, y, z, face, vertex_index, chunk_size)
        ao_values.append(ao_value)
    
    return tuple(ao_values)

def get_simplified_ao(blocks, x, y, z, face, chunk_size=16):
    """
    Simplified AO calculation - just check the 4 edge neighbors for each face
    This is faster and gives good results for most cases
    """
    
    base_ao = 0.8  # Base ambient occlusion value
    
    if face == 'top':
        # Check blocks above the corners
        neighbors = [
            (x, y+1, z),     # Center top
            (x-1, y+1, z),   # Left
            (x+1, y+1, z),   # Right  
            (x, y+1, z-1),   # Back
            (x, y+1, z+1),   # Front
        ]
    elif face == 'bottom':
        neighbors = [
            (x, y-1, z),
            (x-1, y-1, z),
            (x+1, y-1, z),
            (x, y-1, z-1),
            (x, y-1, z+1),
        ]
    elif face == 'front':
        neighbors = [
            (x, y, z+1),
            (x-1, y, z+1),
            (x+1, y, z+1),
            (x, y-1, z+1),
            (x, y+1, z+1),
        ]
    elif face == 'back':
        neighbors = [
            (x, y, z-1),
            (x-1, y, z-1),
            (x+1, y, z-1),
            (x, y-1, z-1),
            (x, y+1, z-1),
        ]
    elif face == 'right':
        neighbors = [
            (x+1, y, z),
            (x+1, y-1, z),
            (x+1, y+1, z),
            (x+1, y, z-1),
            (x+1, y, z+1),
        ]
    elif face == 'left':
        neighbors = [
            (x-1, y, z),
            (x-1, y-1, z),
            (x-1, y+1, z),
            (x-1, y, z-1),
            (x-1, y, z+1),
        ]
    else:
        return base_ao
    
    # Count solid neighbors
    solid_count = 0
    for nx, ny, nz in neighbors:
        if is_block_solid(blocks, nx, ny, nz, chunk_size):
            solid_count += 1
    
    # Reduce AO based on solid neighbors
    ao_reduction = (solid_count / len(neighbors)) * 0.3
    return max(0.4, base_ao - ao_reduction)
