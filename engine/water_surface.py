"""
Water surface rendering system inspired by ornek2
Creates a flat, semi-transparent water plane at the water line
"""

import numpy as np
import moderngl as mgl
import glm
from world.terrain_generator import WATER_LINE

class WaterSurface:
    def __init__(self, renderer):
        self.renderer = renderer
        self.ctx = renderer.ctx
        # Large water area like ornek2: 5 * CHUNK_SIZE * WORLD_W
        # Using our chunk size (16) and a large world multiplier
        self.water_area = 5 * 16 * 50  # 4000 units - covers huge area
        
        # Create water surface mesh
        self.vao = self._create_water_mesh()
        
    def _create_water_mesh(self):
        """Create a large quad mesh for the water surface"""
        # Get water shader program
        water_program = self.renderer.shader_manager.get_program('water')
        
        # Create a large quad at Y = WATER_LINE
        size = self.water_area
        half_size = size // 2
        
        # Vertices for a quad like ornek2 (u, v, x, y, z)
        vertices = np.array([
            # Triangle 1
            [0.0, 0.0, 0.0, 0.0, 0.0],  # Bottom-left
            [1.0, 1.0, 1.0, 0.0, 1.0],  # Top-right  
            [1.0, 0.0, 1.0, 0.0, 0.0],  # Bottom-right
            
            # Triangle 2
            [0.0, 0.0, 0.0, 0.0, 0.0],  # Bottom-left
            [0.0, 1.0, 0.0, 0.0, 1.0],  # Top-left
            [1.0, 1.0, 1.0, 0.0, 1.0],  # Top-right
        ], dtype=np.float32)
        
        # Create VBO
        vbo = self.ctx.buffer(vertices.tobytes())
        
        # Create VAO like ornek2 (tex_coord first, then position)
        vao = self.ctx.vertex_array(water_program, [(vbo, '2f 3f', 'in_tex_coord', 'in_position')])
        
        print("Water surface mesh created successfully")
        return vao
    
    def render(self, view_matrix, proj_matrix, camera_pos):
        """Render the water surface with transparency"""
        # Get water shader program
        water_program = self.renderer.shader_manager.get_program('water')
        
        # Enable blending for transparency
        self.ctx.enable(mgl.BLEND)
        self.ctx.blend_func = mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA
        
        # Disable depth writing but keep depth testing
        self.ctx.depth_mask = False
        
        # Disable face culling for water surface
        if self.ctx.cull_face:
            self.ctx.disable(mgl.CULL_FACE)
            cull_was_enabled = True
        else:
            cull_was_enabled = False
        
        # Bind water texture
        self.renderer.bind_water_texture()
        
        # Set uniforms like ornek2
        water_program['m_view'].write(view_matrix.to_bytes())
        water_program['m_proj'].write(proj_matrix.to_bytes())
        water_program['water_area'] = self.water_area  # Use our calculated water area
        water_program['water_line'] = float(WATER_LINE)
        
        # Render the water surface
        self.vao.render()
        
        # Restore OpenGL state
        self.ctx.depth_mask = True
        self.ctx.disable(mgl.BLEND)
        
        if cull_was_enabled:
            self.ctx.enable(mgl.CULL_FACE)
