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
        # Side length of the plane, in world units. Set from the fog distance by
        # set_fog(); this is just a value to hold until the renderer calls it.
        self.water_area = 200.0

        # Create water surface mesh
        self.vao = self._create_water_mesh()

    def _create_water_mesh(self):
        """Unit quad in XZ; the vertex shader scales and centres it.

        The UV column the mesh used to carry is gone — UVs are now derived from
        world position in the shader so the texture doesn't swim as the plane
        follows the player.
        """
        water_program = self.renderer.shader_manager.get_program('water')

        vertices = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0],
        ], dtype=np.float32)

        vbo = self.ctx.buffer(vertices.tobytes())
        vao = self.ctx.vertex_array(water_program, [(vbo, '3f', 'in_position')])

        print("Water surface mesh created successfully")
        return vao
    
    def upload_static_uniforms(self):
        """Write the water uniforms that only change on a resize.

        The projection, the plane's size and its height are fixed between
        resizes. They used to be re-sent every frame, and `to_bytes()` on the
        projection allocated a fresh buffer each time — the same thing the
        chunk program was already fixed for.
        """
        water_program = self.renderer.shader_manager.get_program('water')
        if not water_program:
            return

        water_program['m_proj'].write(self.renderer.proj_matrix.to_bytes())
        water_program['water_area'] = self.water_area
        water_program['water_line'] = float(WATER_LINE)

    def set_fog(self, fog_range, fog_end):
        """Match the terrain's fog, and shrink the plane to fit inside it.

        2.1 * fog_end across, centred on the player, puts the nearest rim just
        past the distance where fog is already opaque — so the edge of the water
        is never visible, at any render distance.
        """
        water_program = self.renderer.shader_manager.get_program('water')
        if not water_program:
            return

        self.water_area = 2.1 * fog_end
        water_program['water_area'] = self.water_area
        water_program['fog_range'].write(fog_range)

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
        
        # The view matrix and the eye are the only things that move; the rest
        # live in upload_static_uniforms / set_fog.
        water_program['m_view'].write(view_matrix.to_bytes())
        water_program['cam_pos'].write(glm.vec3(camera_pos))
        water_program['water_center'].write(glm.vec2(camera_pos.x, camera_pos.z))

        # Render the water surface
        self.vao.render()
        
        # Restore OpenGL state
        self.ctx.depth_mask = True
        self.ctx.disable(mgl.BLEND)
        
        if cull_was_enabled:
            self.ctx.enable(mgl.CULL_FACE)
