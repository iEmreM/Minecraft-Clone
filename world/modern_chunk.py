import numpy as np
from numba import njit
import glm

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

# Chunk settings
CHUNK_SIZE = 16
CHUNK_HEIGHT = 256

class ModernChunk:
    def __init__(self, chunk_x, chunk_z, renderer, chunk_data=None):
        self.chunk_x = chunk_x
        self.chunk_z = chunk_z
        self.renderer = renderer
        
        # Initialize block data
        self.blocks = np.zeros((CHUNK_SIZE, CHUNK_HEIGHT, CHUNK_SIZE), dtype=np.uint8)
        
        # Rendering data
        self.vao = None
        self.vertex_count = 0
        self.needs_update = True
        
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
            'chunk_z': self.chunk_z
        }
    
    def load_chunk_data(self, chunk_data):
        """Load chunk data from saved state"""
        self.blocks = chunk_data['blocks'].copy()
        self.is_generated = chunk_data.get('is_generated', True)
        self.is_modified = chunk_data.get('is_modified', False)
        self.needs_update = True

    def generate_simple_terrain(self):
        """Generate proper layered terrain"""
        for x in range(CHUNK_SIZE):
            for z in range(CHUNK_SIZE):
                # Calculate world coordinates
                world_x = self.chunk_x * CHUNK_SIZE + x
                world_z = self.chunk_z * CHUNK_SIZE + z
                
                # Generate height with some variation
                base_height = 30
                height_variation = int(3 * np.sin(world_x * 0.1) * np.cos(world_z * 0.1))
                surface_height = base_height + height_variation
                
                # Generate terrain layers from bottom to top
                for y in range(surface_height + 1):
                    if y < surface_height - 3:
                        self.blocks[x, y, z] = STONE  # Deep stone layer
                    elif y < surface_height:
                        self.blocks[x, y, z] = DIRT   # Dirt layer
                    else:
                        self.blocks[x, y, z] = GRASS  # Grass surface
        
        self.needs_update = True
    
    def build_mesh(self):
        """Build mesh using Numba optimized fast builder"""
        if not self.needs_update:
            return
        
        from world.fast_builder import build_chunk_mesh_fast
        
        # Call Numba function
        vertices_array, indices_array = build_chunk_mesh_fast(self.blocks, self.chunk_x, self.chunk_z)
        
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
    
    def is_face_exposed(self, x, y, z, dx, dy, dz):
        """Check if a face is exposed (adjacent block is air or out of bounds)"""
        nx, ny, nz = x + dx, y + dy, z + dz
        
        # Out of bounds = exposed
        if nx < 0 or nx >= CHUNK_SIZE or ny < 0 or ny >= CHUNK_HEIGHT or nz < 0 or nz >= CHUNK_SIZE:
            return True
        
        # Adjacent block is air = exposed
        return self.blocks[nx, ny, nz] == AIR
    
    def get_cube_vertices(self, world_x, world_y, world_z):
        """Generate all vertices for a complete cube using the original algorithm"""
        # Using the same cube_vertices function from main.py
        n = 0.5  # Half size
        x, y, z = world_x + 0.5, world_y + 0.5, world_z + 0.5  # Center the cube
        
        return [
            # Top face
            x-n,y+n,z-n, x-n,y+n,z+n, x+n,y+n,z+n, x+n,y+n,z-n,
            # Bottom face  
            x-n,y-n,z-n, x+n,y-n,z-n, x+n,y-n,z+n, x-n,y-n,z+n,
            # Left face
            x-n,y-n,z-n, x-n,y-n,z+n, x-n,y+n,z+n, x-n,y+n,z-n,
            # Right face
            x+n,y-n,z+n, x+n,y-n,z-n, x+n,y+n,z-n, x+n,y+n,z+n,
            # Front face
            x-n,y-n,z+n, x+n,y-n,z+n, x+n,y+n,z+n, x-n,y+n,z+n,
            # Back face
            x+n,y-n,z-n, x-n,y-n,z-n, x-n,y+n,z-n, x+n,y+n,z-n,
        ]
    
    def get_block_texture_coords(self, block_type, face):
        """Get texture coordinates for a specific block type and face from texture atlas"""
        # Texture atlas is 4x4 (16 sub-textures), each sub-texture is 1/4 of the total texture
        tex_size = 1.0 / 4.0  # Each sub-texture is 1/4 of the atlas
        
        # Define texture atlas coordinates for each block type
        # Based on the original main.py texture definitions
        block_textures = {
            GRASS: {  # GRASS = 1
                'top': (1, 3),      # Grass top texture
                'bottom': (0, 2),   # Dirt texture  
                'sides': (0, 3)     # Grass side texture
            },
            DIRT: {   # DIRT = 2
                'all': (0, 2)       # Dirt texture
            },
            STONE: {  # STONE = 3
                'all': (0, 1)       # Stone texture
            },
            SAND: {   # SAND = 4
                'all': (1, 2)       # Sand texture
            },
            SNOW: {   # SNOW = 5
                'all': (3, 3)       # Snow texture
            },
            LEAVES: { # LEAVES = 6
                'all': (1, 0)       # Leaves texture
            },
            WOOD: {   # WOOD = 7
                'all': (2, 1)       # Wood texture
            },
            WATER: {  # WATER = 8
                'all': (3, 0)       # Water texture (transparent blue)
            }
        }
        
        # Get texture coordinates for this block type
        if block_type in block_textures:
            block_tex = block_textures[block_type]
            
            # Check if block has face-specific textures
            if face in block_tex:
                tex_x, tex_y = block_tex[face]
            elif 'sides' in block_tex and face in ['front', 'back', 'left', 'right']:
                tex_x, tex_y = block_tex['sides']
            elif 'all' in block_tex:
                tex_x, tex_y = block_tex['all']
            else:
                tex_x, tex_y = (0, 0)  # Default texture
        else:
            tex_x, tex_y = (0, 0)  # Default texture for unknown blocks
        
        # Calculate UV coordinates for this sub-texture
        u_min = tex_x * tex_size
        v_min = tex_y * tex_size
        u_max = u_min + tex_size
        v_max = v_min + tex_size
        
        # Return UV coordinates for the four corners of the face
        # Order matches the original main.py: bottom-left, bottom-right, top-right, top-left
        # For side faces, we might need to flip V coordinates to fix sideways texture
        if face in ['front', 'back', 'left', 'right']:
            # Flip V coordinates for side faces to fix orientation
            return [
                (u_min, v_max), (u_max, v_max), (u_max, v_min), (u_min, v_min)
            ]
        else:
            # Normal mapping for top and bottom faces
            return [
                (u_min, v_min), (u_max, v_min), (u_max, v_max), (u_min, v_max)
            ]

    def get_face_vertices(self, world_x, world_y, world_z, face, block_type):
        """Get vertices for a specific face of a block with proper texture atlas mapping"""
        # Get all cube vertices
        cube_verts = self.get_cube_vertices(world_x, world_y, world_z)
        
        # Extract vertices for the specific face
        face_indices = {
            'top': (0, 4),      # vertices 0-3
            'bottom': (4, 8),   # vertices 4-7  
            'left': (8, 12),    # vertices 8-11
            'right': (12, 16),  # vertices 12-15
            'front': (16, 20),  # vertices 16-19
            'back': (20, 24)    # vertices 20-23
        }
        
        if face not in face_indices:
            return []
        
        start, end = face_indices[face]
        face_verts = cube_verts[start*3:end*3]  # *3 because each vertex has 3 coordinates
        
        # Add texture coordinates and shading for each vertex
        vertices = []
        shading_values = {
            'top': 1.0,     # Brightest
            'bottom': 0.4,  # Darkest
            'front': 0.8,   # Medium bright
            'back': 0.8,    # Medium bright  
            'left': 0.6,    # Medium
            'right': 0.6    # Medium
        }
        
        shading = shading_values.get(face, 0.8)
        
        # Get texture coordinates for this block type and face
        tex_coords = self.get_block_texture_coords(block_type, face)
        
        # Calculate ambient occlusion for this face
        from world.ambient_occlusion import get_simplified_ao
        ao_value = get_simplified_ao(self.blocks, world_x, world_y, world_z, face, CHUNK_SIZE)
        
        # Combine base shading with ambient occlusion
        final_shading = shading * ao_value
        
        for i in range(0, len(face_verts), 3):
            x, y, z = face_verts[i], face_verts[i+1], face_verts[i+2]
            # Get texture coordinate for this vertex (4 vertices per face)
            vertex_idx = (i // 3) % 4
            u, v = tex_coords[vertex_idx]
            vertices.extend([x, y, z, u, v, final_shading])
        
        return vertices
    
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
    
    def render(self):
        """Render this chunk"""
        if self.needs_update:
            self.build_mesh()
        
        if self.vao and self.vertex_count > 0:
            self.renderer.render_vao(self.vao)
    

