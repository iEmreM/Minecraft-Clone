import numpy as np

# Block types
AIR = 0
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

# Chunk settings
CHUNK_SIZE = 16
CHUNK_HEIGHT = 256

class ModernChunk:
    def __init__(self, chunk_x, chunk_z, renderer, chunk_data=None, chunk_manager=None):
        self.chunk_x = chunk_x
        self.chunk_z = chunk_z
        self.renderer = renderer
        self.chunk_manager = chunk_manager  # Reference to ThreadedChunkManager for async mesh requests
        
        # Initialize block data
        self.blocks = np.zeros((CHUNK_SIZE, CHUNK_HEIGHT, CHUNK_SIZE), dtype=np.uint8)
        
        # Rendering data
        self.vao = None
        self.vertex_count = 0
        self.needs_update = True
        
        # Mesh cache for performance optimization
        self.cached_vertices = None
        self.cached_indices = None
        self.mesh_cache_valid = False
        
        # Persistence tracking
        self.is_generated = False
        self.is_modified = False
        
        # Load existing chunk data or generate new terrain
        if chunk_data is not None:
            self.load_chunk_data(chunk_data)
        else:
            self.generate_advanced_terrain()
    
    def generate_advanced_terrain(self):
        """Generate advanced terrain using the new terrain generator"""
        from world.terrain_generator import terrain_generator
        terrain_generator.generate_chunk_terrain(self.chunk_x, self.chunk_z, self.blocks)
        self.is_generated = True
        self.needs_update = True
    
    def save_chunk_data(self):
        """Save chunk data for persistence"""
        return {
            'blocks': self.blocks.copy(),
            'is_generated': self.is_generated,
            'is_modified': self.is_modified,
            'chunk_x': self.chunk_x,
            'chunk_z': self.chunk_z,
            # Save mesh cache for faster reloading
            'cached_vertices': self.cached_vertices,
            'cached_indices': self.cached_indices,
            'mesh_cache_valid': self.mesh_cache_valid
        }
    
    def load_chunk_data(self, chunk_data):
        """Load chunk data from saved state"""
        self.blocks = chunk_data['blocks'].copy()
        self.is_generated = chunk_data.get('is_generated', True)
        self.is_modified = chunk_data.get('is_modified', False)
        # Restore mesh cache if available
        self.cached_vertices = chunk_data.get('cached_vertices', None)
        self.cached_indices = chunk_data.get('cached_indices', None)
        self.mesh_cache_valid = chunk_data.get('mesh_cache_valid', False)
        self.needs_update = True

    def build_mesh(self):
        """Build mesh using Numba optimized fast builder"""
        if not self.needs_update:
            return
        
        from world.fast_builder import build_chunk_mesh_fast
        
        # Check if we have a valid cached mesh and can skip regeneration
        if self.mesh_cache_valid and self.cached_vertices is not None and self.cached_indices is not None:
            # Reuse cached mesh data - only recreate VAO
            vertices_array = self.cached_vertices
            indices_array = self.cached_indices
        else:
            # Generate new mesh using Numba function
            vertices_array, indices_array = build_chunk_mesh_fast(self.blocks, self.chunk_x, self.chunk_z)
            
            # Cache the mesh data for future use
            self.cached_vertices = vertices_array
            self.cached_indices = indices_array
            self.mesh_cache_valid = True
        
        # Create VAO if we have vertices
        if len(vertices_array) > 0:
            # Clean up old VAO
            if self.vao:
                self.vao.release()
            
            # Create new VAO
            self.vao = self.renderer.create_vao(vertices_array, indices_array)
            self.vertex_count = len(indices_array)
            # print(f"Chunk ({self.chunk_x},{self.chunk_z}): Generated {len(vertices_array)//6} vertices, {len(indices_array)} indices")
        
        self.needs_update = False
    
    def get_block(self, x, y, z):
        """Get block type at local chunk coordinates"""
        if x < 0 or x >= CHUNK_SIZE or y < 0 or y >= CHUNK_HEIGHT or z < 0 or z >= CHUNK_SIZE:
            return AIR
        return self.blocks[x, y, z]
    
    def set_block(self, x, y, z, block_type):
        """Set block type at local chunk coordinates"""
        if x < 0 or x >= CHUNK_SIZE or y < 0 or y >= CHUNK_HEIGHT or z < 0 or z >= CHUNK_SIZE:
            return
        
        old_block = self.blocks[x, y, z]
        if old_block != block_type:
            self.blocks[x, y, z] = block_type
            self.needs_update = True  # Mark chunk for mesh rebuild
            self.is_modified = True   # Mark chunk as modified by player
            self.mesh_cache_valid = False  # Invalidate cache since blocks changed
            
            # Request async mesh rebuild if chunk_manager is available
            if self.chunk_manager:
                priority = self.chunk_manager._calculate_chunk_priority(self.chunk_x, self.chunk_z)
                self.chunk_manager.mesh_request_counter += 1
                self.chunk_manager.mesh_build_queue.put((priority, self.chunk_manager.mesh_request_counter, {
                    'coords': (self.chunk_x, self.chunk_z),
                    'chunk': self
                }))
    
    def render(self):
        """Render this chunk"""
        # Don't build mesh here anymore - it's built asynchronously in ThreadedChunkManager
        # Just render the VAO if it exists
        if self.vao and self.vertex_count > 0:
            self.renderer.render_vao(self.vao)
    

