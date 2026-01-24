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
# from numba import njit  # Temporarily disable numba for compatibility
from opensimplex import noise2, noise3
# Import constants locally to avoid circular imports
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
TREE_PROBABILITY = 0.03  # Doubled from 0.02 for more trees
TREE_HEIGHT = 6
TREE_WIDTH = 2

def get_terrain_height(world_x, world_z):
    """
    Generate terrain height using simplified version without extreme island mask
    """
    x, z = world_x, world_z
    
    # Amplitude (from ornek2)
    a1 = CENTER_Y
    a2, a4, a8 = a1 * 0.5, a1 * 0.25, a1 * 0.125
    
    # Frequency (from ornek2)
    f1 = 0.005
    f2, f4, f8 = f1 * 2, f1 * 4, f1 * 8
    
    # Terrain variation (from ornek2)
    if noise2(0.1 * x, 0.1 * z) < 0:
        a1 /= 1.07
    
    height = 0
    height += noise2(x * f1, z * f1) * a1 + a1
    height += noise2(x * f2, z * f2) * a2 - a2
    height += noise2(x * f4, z * f4) * a4 + a4
    height += noise2(x * f8, z * f8) * a8 - a8
    
    height = max(height, noise2(x * f8, z * f8) + 2)
    
    # Apply much gentler island mask for near-spawn area
    distance = math.sqrt((x - WORLD_CENTER) ** 2 + (z - WORLD_CENTER) ** 2)
    if distance > 2000:  # Only apply island mask far from center
        island_factor = max(0.1, 1.0 - (distance - 2000) / 3000)
        height *= island_factor
    
    return int(max(height, 8))  # Ensure minimum height

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
        if (noise3(world_x * 0.09, world_y * 0.09, world_z * 0.09) > 0 and
            noise2(world_x * 0.1, world_z * 0.1) * 3 + 3 < world_y < terrain_height - 10):
            return AIR
        else:
            return STONE
    else:
        # Surface layer with variation (from ornek2)
        import random
        rng = int(7 * random.random())
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

def is_cave(world_x, world_y, world_z, terrain_height):
    """
    Check if this position should be a cave using 3D noise
    """
    # Only generate caves underground, not too deep, not too shallow
    if world_y <= CAVE_MIN_Y or world_y >= terrain_height - CAVE_MAX_Y_OFFSET:
        return False
    
    # Use 3D noise to create cave systems
    cave_noise1 = noise3(world_x * CAVE_FREQUENCY, world_y * CAVE_FREQUENCY, world_z * CAVE_FREQUENCY)
    cave_noise2 = noise2(world_x * 0.08, world_z * 0.08)
    
    # Create caves where noise exceeds threshold
    return cave_noise1 > CAVE_THRESHOLD and cave_noise2 * 3 + 3 < world_y

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
    
    # Use random like ornek2
    import random
    rnd = random.random()
    return rnd <= TREE_PROBABILITY

def generate_tree(blocks, local_x, local_y, local_z, chunk_size=CHUNK_SIZE):
    """
    Generate a tree structure using ornek2's algorithm
    """
    TREE_H_WIDTH = TREE_WIDTH // 2
    TREE_H_HEIGHT = TREE_HEIGHT // 2
    
    # Check bounds (from ornek2)
    if local_y + TREE_HEIGHT >= CHUNK_HEIGHT:
        return
    if local_x - TREE_H_WIDTH < 0 or local_x + TREE_H_WIDTH >= chunk_size:
        return
    if local_z - TREE_H_WIDTH < 0 or local_z + TREE_H_WIDTH >= chunk_size:
        return
    
    # Dirt under the tree (from ornek2)
    blocks[local_x, local_y, local_z] = DIRT
    
    # Simple cube-based leaves generation with density decreasing upward
    import random
    
    # Define leaves area - cube around the top of the tree
    leaves_size = 2  # Creates a 5x5 cube (2 blocks in each direction from center)
    leaves_bottom_y = local_y + TREE_HEIGHT - 3  # Start leaves 3 blocks from tree top
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
                if 0 <= leaves_x < chunk_size and 0 <= leaves_z < chunk_size:
                    # Don't replace tree trunk (center column)
                    if ix == 0 and iz == 0:
                        continue
                    
                    # Place leaves based on density
                    if random.random() < density:
                        blocks[leaves_x, current_y, leaves_z] = LEAVES
    
    # Add random side extensions to make tree more natural
    for _ in range(8):  # 8 random side extensions
        # Pick a random direction
        direction = random.choice([(1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,-1), (1,-1), (-1,1)])
        extend_x = local_x + direction[0] * (leaves_size + 1)
        extend_z = local_z + direction[1] * (leaves_size + 1)
        extend_y = leaves_bottom_y + random.randint(0, 2)  # Random height in lower layers
        
        if (0 <= extend_x < chunk_size and 0 <= extend_z < chunk_size and 
            extend_y < CHUNK_HEIGHT):
            blocks[extend_x, extend_y, extend_z] = LEAVES
    
    # Tree trunk (from ornek2)
    for iy in range(1, TREE_HEIGHT - 2):
        trunk_y = local_y + iy
        if trunk_y < CHUNK_HEIGHT:
            blocks[local_x, trunk_y, local_z] = WOOD
    
    # Top (from ornek2)
    top_y = local_y + TREE_HEIGHT - 2
    if top_y < CHUNK_HEIGHT:
        blocks[local_x, top_y, local_z] = LEAVES

class AdvancedTerrainGenerator:
    """
    Advanced terrain generator using multi-octave noise
    """
    
    def __init__(self, seed=WORLD_SEED):
        self.seed = seed
        # Initialize noise with seed
        np.random.seed(seed)
    
    def generate_chunk_terrain(self, chunk_x, chunk_z, blocks):
        """
        Generate terrain for a chunk using advanced algorithms
        """
        # Clear the blocks array
        blocks.fill(AIR)
        
        # Generate terrain
        for local_x in range(CHUNK_SIZE):
            for local_z in range(CHUNK_SIZE):
                # Calculate world coordinates
                world_x = chunk_x * CHUNK_SIZE + local_x
                world_z = chunk_z * CHUNK_SIZE + local_z
                
                # Get terrain height for this column
                terrain_height = get_terrain_height(world_x, world_z)
                
                # Generate vertical column
                for local_y in range(min(terrain_height + 1, CHUNK_HEIGHT)):
                    world_y = local_y  # Assuming chunks start at Y=0
                    
                    # Determine block type
                    block_type = get_block_type(world_x, world_y, world_z, terrain_height)
                    blocks[local_x, local_y, local_z] = block_type
                    
                    # Check for tree placement on surface
                    if (local_y == terrain_height and block_type == GRASS and
                        should_place_tree(world_x, world_y, world_z, block_type)):
                        generate_tree(blocks, local_x, local_y, local_z)

# Global terrain generator instance
terrain_generator = AdvancedTerrainGenerator()
