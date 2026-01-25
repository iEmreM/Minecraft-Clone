import numpy as np
import moderngl as mgl
import glm

class SkyRenderer:
    def __init__(self, ctx, shader_manager):
        self.ctx = ctx
        self.shader_manager = shader_manager
        
        # Load sky shader
        self.shader_manager.load_shader('sky', 'shaders/sky.vert', 'shaders/sky.frag')
        self.program = self.shader_manager.get_program('sky')
        
        # Fullscreen Quad Geometry (2 Triangles, covering -1 to 1 clip space)
        # z=0.9999 to be at the far plane (behind everything)
        # We only need X, Y coordinates, Z can be inferred or set in shader
        self.vertices = np.array([
            -1.0, -1.0, 0.0,
             1.0, -1.0, 0.0,
            -1.0,  1.0, 0.0,
             1.0,  1.0, 0.0,
        ], dtype=np.float32)
        
        self.vbo = self.ctx.buffer(self.vertices.tobytes())
        
        # Layout: 2 floats for position? No, I defined vec3 in shader.
        # But shader uses layout location=1 for position, 0 for texcoord?
        # My shader `sky.vert` uses `in_position` at location 1 (vec3).
        # Let's fix shader to generic layout or match this.
        # Actually sky.vert expects:
        # layout (location = 1) in vec3 in_position;
        # layout (location = 0) in vec2 in_tex_coord; (Unused in main but present)
        # Let's just pass vec3 position.
        
        self.vao = self.ctx.vertex_array(
            self.program,
            [(self.vbo, '3f', 'in_position')]
        )
        
    def render(self, view_matrix, proj_matrix, time):
        if not self.program:
            return
            
        self.ctx.disable(mgl.DEPTH_TEST) # Sky is background, no depth test needed if drawn first
        # OR: self.ctx.enable(mgl.DEPTH_TEST) and gl_FragDepth = 1.0
        # Drawing first with Depth Test DISABLED is easiest for skybox behavior
        
        # Calculate Inverse View-Projection Matrix for Ray Re-projection
        # We need this to calculate world/view direction from screen pixels
        
        # View matrix rotation only (remove translation) for skybox to look infinite
        view_rot = glm.mat4(glm.mat3(view_matrix))
        inv_pv = glm.inverse(proj_matrix * view_rot)
        
        self.program['m_inv_pv'].write(inv_pv.to_bytes())
        self.program['u_time'] = time
        
        # Draw quad
        self.vao.render(mgl.TRIANGLE_STRIP)
        
        self.ctx.enable(mgl.DEPTH_TEST) # Restore depth test
