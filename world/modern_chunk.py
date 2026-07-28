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
        
        # Rendering data. The chunk owns all three GL objects — see release_gl.
        self.vao = None
        self.vbo = None
        self.ibo = None
        self.vertex_count = 0

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
    
    def save_chunk_data(self):
        """Save chunk data for persistence.

        Block data only — the mesh is not cached. A mesh is built against the
        neighbors that happened to be loaded at the time, so a stored one goes
        stale as soon as the surroundings differ, and rebuilding it costs ~1 ms.
        """
        return {
            'blocks': self.blocks.copy(),
            'is_generated': self.is_generated,
            'is_modified': self.is_modified,
            'chunk_x': self.chunk_x,
            'chunk_z': self.chunk_z,
        }

    def load_chunk_data(self, chunk_data):
        """Load chunk data from saved state"""
        self.blocks = chunk_data['blocks'].copy()
        self.is_generated = chunk_data.get('is_generated', True)
        self.is_modified = chunk_data.get('is_modified', False)

    def upload_mesh(self, vertices, indices):
        """Hand a freshly built mesh to the GPU. Main thread only (OpenGL)."""
        self.release_gl()

        if len(vertices) > 0:
            self.vao, self.vbo, self.ibo = self.renderer.create_vao(vertices, indices)
            self.vertex_count = len(indices)

    def release_gl(self):
        """Free this chunk's GPU objects.

        moderngl's gc_mode defaults to None, so nothing is collected on its own,
        and VertexArray.release() frees only the VAO — the VBO and IBO behind it
        stay allocated. Releasing them by hand is the whole fix: without it every
        chunk unload and every mesh rebuild leaked the chunk's whole mesh,
        ~180 KB for average terrain.
        """
        for gl_object in (self.vao, self.vbo, self.ibo):
            if gl_object is not None:
                gl_object.release()

        self.vao = None
        self.vbo = None
        self.ibo = None
        self.vertex_count = 0

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
            self.is_modified = True   # Mark chunk as modified by player

            # Request async mesh rebuild if chunk_manager is available
            if self.chunk_manager:
                self.chunk_manager.request_mesh(self.chunk_x, self.chunk_z, self)

                # A block on the seam changes what the chunk next door can see
                # through it, so that one has to be rebuilt as well — otherwise
                # it keeps the face it meshed against the old block.
                if x == 0:
                    self.chunk_manager.request_mesh(self.chunk_x - 1, self.chunk_z)
                elif x == CHUNK_SIZE - 1:
                    self.chunk_manager.request_mesh(self.chunk_x + 1, self.chunk_z)
                if z == 0:
                    self.chunk_manager.request_mesh(self.chunk_x, self.chunk_z - 1)
                elif z == CHUNK_SIZE - 1:
                    self.chunk_manager.request_mesh(self.chunk_x, self.chunk_z + 1)
    
    def render(self):
        """Render this chunk"""
        # Don't build mesh here anymore - it's built asynchronously in ThreadedChunkManager
        # Just render the VAO if it exists
        if self.vao and self.vertex_count > 0:
            self.renderer.render_vao(self.vao)
    

