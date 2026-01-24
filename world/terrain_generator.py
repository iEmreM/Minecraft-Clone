"""
Advanced terrain generation system inspired by ornek2
Features:
- Multi-octave Perlin noise for realistic terrain
- Biome-based generation with different elevation levels  
- Cave generation using 3D noise
- Tree placement with realistic structures
- Multiple block types (grass, dirt, stone, sand, snow)
"""

import numpy as np
import math
from numba import njit
from world.fast_noise import fast_noise2, fast_noise3, seed_noise

# Block type constants
AIR = 0
GRASS = 1
DIRT = 2
STONE = 3
SAND = 4
SNOW = 5
LEAVES = 6
WOOD = 7
WATER = 8

# Chunk constants
CHUNK_SIZE = 16
CHUNK_HEIGHT = 256

# World generation settings
WORLD_SEED = 42
CENTER_Y = 48  # Base terrain height like ornek2
WORLD_CENTER = 480  # World center for island generation
WATER_LINE = 11.95  # Water level like ornek2

# Terrain levels (Y coordinates) - from ornek2
SNOW_LEVEL = 54
STONE_LEVEL = 49  
DIRT_LEVEL = 40
GRASS_LEVEL = 11
SAND_LEVEL = 5

# Noise frequencies for different octaves
FREQ_1 = 0.005   # Large terrain features
FREQ_2 = 0.01    # Medium features  
FREQ_4 = 0.02    # Small features
FREQ_8 = 0.04    # Fine details

# Amplitudes for different octaves
AMP_1 = CENTER_Y
AMP_2 = AMP_1 * 0.5
AMP_4 = AMP_1 * 0.25
AMP_8 = AMP_1 * 0.125

# Cave generation
CAVE_FREQUENCY = 0.05
CAVE_THRESHOLD = 0.0
CAVE_MIN_Y = 5
CAVE_MAX_Y_OFFSET = 10

# Tree generation
TREE_PROBABILITY = 0.03
TREE_HEIGHT = 6
TREE_WIDTH = 2
TREE_H_WIDTH = 1

@njit
def fast_rand(x, y, z):
    """Deterministic random float between 0.0 and 1.0 based on coordinates"""
    # Simple hash based random
    n = int(x * 374761393 + y * 668265263 + z * 437585453)
    n = (n ^ (n >> 13)) * 1274126177
    return ((n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff) / 2147483647.0

@njit
def get_terrain_height(world_x, world_z):
    """
    Generate terrain height using simplified version without extreme island mask
    """
    x, z = float(world_x), float(world_z)
    
    # Amplitude (from ornek2)
    a1 = float(CENTER_Y)
    a2, a4, a8 = a1 * 0.5, a1 * 0.25, a1 * 0.125
    
    # Frequency (from ornek2)
    f1 = 0.005
    f2, f4, f8 = f1 * 2, f1 * 4, f1 * 8
    
    # Terrain variation (from ornek2)
    if fast_noise2(0.1 * x, 0.1 * z) < 0:
        a1 /= 1.07
    
    height = 0.0
    height += fast_noise2(x * f1, z * f1) * a1 + a1
    height += fast_noise2(x * f2, z * f2) * a2 - a2
    height += fast_noise2(x * f4, z * f4) * a4 + a4
    height += fast_noise2(x * f8, z * f8) * a8 - a8
    
    height = max(height, fast_noise2(x * f8, z * f8) + 2.0)
    
    # Apply much gentler island mask for near-spawn area
    dx = x - WORLD_CENTER
    dz = z - WORLD_CENTER
    distance = math.sqrt(dx*dx + dz*dz)
    
    if distance > 2000.0:  # Only apply island mask far from center
        island_factor = max(0.1, 1.0 - (distance - 2000.0) / 3000.0)
        height *= island_factor
    
    return int(max(height, 8.0))

@njit
def get_block_type(world_x, world_y, world_z, terrain_height):
    """
    Determine block type based on position and terrain height (from ornek2)
    """
    if world_y > terrain_height:
        # Add water at low levels
        if world_y <= WATER_LINE:
            return WATER
        return AIR
    
    # Underground structure (from ornek2)
    if world_y < terrain_height - 1:
        # Create caves (from ornek2)
        # Using 3D noise
        cave_n1 = fast_noise3(world_x * 0.09, world_y * 0.09, world_z * 0.09)
        cave_n2 = fast_noise2(world_x * 0.1, world_z * 0.1)
        
        if (cave_n1 > 0 and cave_n2 * 3 + 3 < world_y < terrain_height - 10):
            return AIR
        else:
            return STONE
    else:
        # Surface layer with variation (from ornek2)
        # Random falloff
        rng = int(7 * fast_rand(world_x, world_y, world_z))
        ry = world_y - rng
        
        # Fixed logic: check from highest to lowest, with proper fallbacks
        if ry >= SNOW_LEVEL:
            return SNOW  # High mountains get snow
        elif ry >= STONE_LEVEL:
            return STONE  # High elevation gets stone
        elif ry >= DIRT_LEVEL:
            return DIRT   # Medium elevation gets dirt
        elif ry >= GRASS_LEVEL:
            return GRASS  # Low-medium elevation gets grass
        elif ry >= SAND_LEVEL:
            return SAND   # Low elevation gets sand
        else:
            return STONE  # Very low/underground defaults to stone

@njit
def should_place_tree(world_x, world_y, world_z, block_type):
    """
    Determine if a tree should be placed at this location (from ornek2)
    """
    # Only place trees on grass and below dirt level
    if block_type != GRASS or world_y >= DIRT_LEVEL:
        return False
    
    # Don't place trees underwater (terrain must be above water line)
    if world_y <= WATER_LINE:
        return False
    
    # Use spatial hash for random
    rnd = fast_rand(world_x, world_y, world_z)
    return rnd <= TREE_PROBABILITY

@njit
def generate_tree_fast(blocks, local_x, local_y, local_z):
    """
    Generate a tree structure (Numba optimized)
    """
    TREE_H_WIDTH_LOCAL = 1
    TREE_HEIGHT_LOCAL = 6
    
    # Check bounds
    if local_y + TREE_HEIGHT_LOCAL >= CHUNK_HEIGHT:
        return
    if local_x - TREE_H_WIDTH_LOCAL < 0 or local_x + TREE_H_WIDTH_LOCAL >= CHUNK_SIZE:
        return
    if local_z - TREE_H_WIDTH_LOCAL < 0 or local_z + TREE_H_WIDTH_LOCAL >= CHUNK_SIZE:
        return
    
    # Dirt under the tree (from ornek2)
    blocks[local_x, local_y, local_z] = DIRT
    
    # Define leaves area - cube around the top of the tree
    leaves_size = 2  # Creates a 5x5 cube (2 blocks in each direction from center)
    leaves_bottom_y = local_y + TREE_HEIGHT_LOCAL - 3  # Start leaves 3 blocks from tree top
    leaves_height = 4  # 4 layers of leaves
    
    for layer in range(leaves_height):
        current_y = leaves_bottom_y + layer
        if current_y >= CHUNK_HEIGHT:
            break
            
        # Density decreases as we go up (more leaves at bottom)
        density = 1.0 - (layer * 0.2)  # 100%, 80%, 60%, 40% density per layer
        
        # Generate cube of leaves for this layer
        for ix in range(-leaves_size, leaves_size + 1):
            for iz in range(-leaves_size, leaves_size + 1):
                leaves_x = local_x + ix
                leaves_z = local_z + iz
                
                # Check bounds
                if 0 <= leaves_x < CHUNK_SIZE and 0 <= leaves_z < CHUNK_SIZE:
                    # Don't replace tree trunk (center column)
                    if ix == 0 and iz == 0:
                        continue
                    
                    # Place leaves based on density
                    # Pseudo-random check
                    rng = fast_rand(leaves_x, current_y, leaves_z)
                    if rng < density:
                        blocks[leaves_x, current_y, leaves_z] = LEAVES
    
    # Add random side extensions to make tree more natural
    # We use a fixed deterministic loop instead of random.choice/range
    for i in range(8):  
        # Mock random direction using hash
        h = fast_rand(local_x + i, local_y, local_z + i)
        direction_idx = int(h * 8) % 8
        
        dx = 0
        dz = 0
        if direction_idx == 0: dx=1; dz=0
        elif direction_idx == 1: dx=-1; dz=0
        elif direction_idx == 2: dx=0; dz=1
        elif direction_idx == 3: dx=0; dz=-1
        elif direction_idx == 4: dx=1; dz=1
        elif direction_idx == 5: dx=-1; dz=-1
        elif direction_idx == 6: dx=1; dz=-1
        elif direction_idx == 7: dx=-1; dz=1
        
        extend_x = local_x + dx * (leaves_size + 1)
        extend_z = local_z + dz * (leaves_size + 1)
        
        # Random height offset 0-2
        h2 = fast_rand(extend_x, local_y, extend_z)
        offset_y = int(h2 * 3)
        extend_y = leaves_bottom_y + offset_y
        
        if (0 <= extend_x < CHUNK_SIZE and 0 <= extend_z < CHUNK_SIZE and 
            extend_y < CHUNK_HEIGHT):
            blocks[extend_x, extend_y, extend_z] = LEAVES
    
    # Tree trunk
    for iy in range(1, TREE_HEIGHT_LOCAL - 2):
        trunk_y = local_y + iy
        if trunk_y < CHUNK_HEIGHT:
            blocks[local_x, trunk_y, local_z] = WOOD
    
    # Top
    top_y = local_y + TREE_HEIGHT_LOCAL - 2
    if top_y < CHUNK_HEIGHT:
        blocks[local_x, top_y, local_z] = LEAVES

@njit
def generate_chunk_fast(chunk_x, chunk_z, blocks):
    """
    Main chunk generation function (JIT compiled)
    """
    for lx in range(CHUNK_SIZE):
        for lz in range(CHUNK_SIZE):
            wx = chunk_x * CHUNK_SIZE + lx
            wz = chunk_z * CHUNK_SIZE + lz
            
            h = get_terrain_height(wx, wz)
            
            # Reduce loop range to valid terrain
            # But we must fill everything to be block_type
            # However, blocks is pre-filled with AIR (0).
            # So we only iterate up to h?
            # But get_block_type also handles caves underground.
            
            # The original code iterated up to min(h+1, CHUNK_HEIGHT)
            for ly in range(min(h + 1, CHUNK_HEIGHT)):
                wy = ly
                b_type = get_block_type(wx, wy, wz, h)
                blocks[lx, ly, lz] = b_type
                
                # Tree check
                if ly == h and b_type == GRASS:
                    if should_place_tree(wx, wy, wz, b_type):
                         generate_tree_fast(blocks, lx, ly, lz)

class AdvancedTerrainGenerator:
    """
    Advanced terrain generator using multi-octave noise (Numba Optimized)
    """
    
    def __init__(self, seed=WORLD_SEED):
        self.seed = seed
        # Initialize noise with seed (Not actually used in fast_noise yet, but ready)
        # np.random.seed(seed)
        
        # Warmup JIT (Optional but good for first frame spike prevention)
        # We can call the function with dummy data
        print("Warming up Terrain Generator JIT...")
        dummy = np.zeros((16, 256, 16), dtype=np.uint8)
        generate_chunk_fast(0, 0, dummy)
        print("Terrain Generator JIT Ready.")
    
    def generate_chunk_terrain(self, chunk_x, chunk_z, blocks):
        """
        Generate terrain for a chunk using advanced algorithms
        """
        # Clear the blocks array
        blocks.fill(AIR)
        
        # Generate terrain using Numba function
        generate_chunk_fast(chunk_x, chunk_z, blocks)

# Global terrain generator instance
terrain_generator = AdvancedTerrainGenerator()
