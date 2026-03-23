import moderngl as mgl
import pygame as pg
import numpy as np
from engine.shader_manager import ShaderManager
from engine.water_surface import WaterSurface
from engine.sky import SkyRenderer
import glm
import math


class ModernGLRenderer:
    def __init__(self, width=800, height=600):
        # Initialize Pygame
        pg.init()
        
        # Set OpenGL attributes for modern context
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE)
        pg.display.gl_set_attribute(pg.GL_DEPTH_SIZE, 24)
        
        # Enable MSAA (Anti-Aliasing)
        pg.display.gl_set_attribute(pg.GL_MULTISAMPLEBUFFERS, 1)
        pg.display.gl_set_attribute(pg.GL_MULTISAMPLESAMPLES, 4)
        
        # Create display
        self.screen = pg.display.set_mode((width, height), pg.OPENGL | pg.DOUBLEBUF | pg.RESIZABLE)
        pg.display.set_caption('Minecraft Clone - ModernGL')
        
        # Create ModernGL context
        self.ctx = mgl.create_context()
        # Enable depth testing and face culling like ornek2
        self.ctx.enable(mgl.DEPTH_TEST | mgl.CULL_FACE)
        self.ctx.cull_face = 'back'
        
        # Initialize shader manager
        self.shader_manager = ShaderManager(self.ctx)
        self.shader_manager.load_default_shaders()
        
        # Get chunk shader program
        self.chunk_program = self.shader_manager.get_program('chunk')
        
        # Store screen dimensions
        self.width = width
        self.height = height
        
        # Initialize textures
        self.block_texture = None
        self.water_texture = None
        
        # Initialize water surface
        self.water_surface = WaterSurface(self)
        
        # Initialize sky renderer
        self.sky_renderer = SkyRenderer(self.ctx, self.shader_manager)
        
        # Create projection matrix with proper far distance like ornek2
        self.proj_matrix = glm.perspective(
            glm.radians(65.0), 
            width / height, 
            0.1, 
            1000.0  # Use far distance like ornek2 for proper render distance
        )
        
        # Initialize crosshair and block outline
        self._init_crosshair()
        self._init_block_outline()
        
        print("ModernGL Renderer initialized successfully")
    
    def resize(self, width, height):
        """Handle window resize"""
        self.width = width
        self.height = height
        self.ctx.viewport = (0, 0, width, height)
        
        # Update projection matrix with increased render distance
        self.proj_matrix = glm.perspective(
            glm.radians(65.0),
            width / height,
            0.1,
            5000.0  # Increased from 100 to 500 for better fog visibility
        )
    
    def clear(self):
        """Clear the screen with sky background color like ornek2"""
        # Set sky color to match procedural sky horizon
        self.bg_color = glm.vec3(0.6, 0.8, 0.95)
        self.ctx.clear(color=(self.bg_color.x, self.bg_color.y, self.bg_color.z))
    
    def set_view_matrix(self, view_matrix):
        """Set the view matrix for rendering"""
        if self.chunk_program:
            self.chunk_program['m_view'].write(view_matrix.to_bytes())
    
    def set_model_matrix(self, model_matrix):
        """Set the model matrix for rendering"""
        # OPTIMIZATION: m_model removed - shader compiler optimizes it out
        # Keeping this function for API compatibility but it does nothing now
        pass
    
    def update_matrices(self, view_matrix, model_matrix=None):
        """Update projection, view, and model matrices"""
        if not self.chunk_program:
            return
            
        # Set projection matrix
        self.chunk_program['m_proj'].write(self.proj_matrix.to_bytes())
        
        # Set view matrix
        self.chunk_program['m_view'].write(view_matrix.to_bytes())
        
        # OPTIMIZATION: m_model removed - shader compiler optimizes it out since unused
        # The vertex shader doesn't use m_model (always identity), so GLSL compiler removes it
        
        # Set background color for fog effect
        if hasattr(self, 'bg_color'):
            self.chunk_program['bg_color'].write(self.bg_color)
        
        # Set water line for underwater effects (from ornek2)
        from world.terrain_generator import WATER_LINE
        self.chunk_program['water_line'] = float(WATER_LINE)
    
    def create_vao(self, vertices, indices=None):
        """Create a Vertex Array Object from vertex data"""
        if vertices.size == 0:
            return None
            
        # Create vertex buffer
        vbo = self.ctx.buffer(vertices.astype(np.float32).tobytes())
        
        # Create VAO
        if indices is not None:
            # Create index buffer
            ibo = self.ctx.buffer(indices.astype(np.uint32).tobytes())
            # Updated format: 3f position, 3f tex_coord (vec3), 1f shading
            vao = self.ctx.vertex_array(self.chunk_program, [(vbo, '3f 3f 1f', 'in_position', 'in_tex_coord', 'in_shading')], ibo)
        else:
            vao = self.ctx.vertex_array(self.chunk_program, [(vbo, '3f 3f 1f', 'in_position', 'in_tex_coord', 'in_shading')])
        
        return vao
    
    def render_vao(self, vao):
        """Render a Vertex Array Object"""
        if vao and self.chunk_program:
            vao.render()
    
    def create_texture_array(self, texture_path, tile_count_x=4, tile_count_y=4):
        """Create a texture array from an atlas"""
        try:
            # Load texture using pygame
            texture_surface = pg.image.load(texture_path).convert_alpha()  # Always use alpha for consistency
            
            width = texture_surface.get_width()
            height = texture_surface.get_height()
            
            tile_width = width // tile_count_x
            tile_height = height // tile_count_y
            
            # Extract sub-images
            layers = []
            for y in range(tile_count_y):
                for x in range(tile_count_x):
                    # Get sub-surface
                    rect = pg.Rect(x * tile_width, y * tile_height, tile_width, tile_height)
                    sub_surface = texture_surface.subsurface(rect)
                    
                    # Convert to string buffer
                    data = pg.image.tostring(sub_surface, 'RGBA') # 4 components
                    layers.append(data)
            
            # Combine all layers into one bytes object
            full_data = b''.join(layers)
            
            # Create Texture Array
            # Size: (width, height, layers)
            texture_array = self.ctx.texture_array(
                (tile_width, tile_height, len(layers)),
                4, # RGBA
                full_data
            )
            
            # Set parameters
            texture_array.filter = (mgl.NEAREST_MIPMAP_NEAREST, mgl.NEAREST)
            texture_array.repeat_x = True # Allow tiling
            texture_array.repeat_y = True
            texture_array.build_mipmaps()
            
            return texture_array
            
        except Exception as e:
            print(f"Error creating texture array: {e}")
            return None
    
    def create_texture(self, texture_path, components=3, has_alpha=False):
        """Create a texture from an image file (kept for non-array textures like water)"""
        try:
            # Load texture using pygame
            texture_surface = pg.image.load(texture_path)
            
            # Handle alpha channel
            if has_alpha:
                texture_data = pg.image.tostring(texture_surface, 'RGBA')
                components = 4
            else:
                texture_data = pg.image.tostring(texture_surface, 'RGB')
                components = 3
            
            # Create ModernGL texture
            texture = self.ctx.texture(
                (texture_surface.get_width(), texture_surface.get_height()),
                components,
                texture_data
            )
            
            # Set texture parameters for pixelated look
            texture.filter = (mgl.NEAREST, mgl.NEAREST)
            texture.repeat_x = True  # Allow tiling for water
            texture.repeat_y = True
            
            return texture
            
        except Exception as e:
            print(f"Error creating texture: {e}")
            return None
    
    def load_textures(self):
        """Load block and water textures"""
        # Load block texture as Array (texture unit 0)
        # Assuming texture.png is 4x4 atlas
        self.block_texture = self.create_texture_array('texture.png', 4, 4)
        if self.block_texture:
            self.block_texture.use(0)
            # Force repeat
            self.block_texture.repeat_x = True
            self.block_texture.repeat_y = True
            print("Block texture array loaded on unit 0 (Repeat: ON)")
        
        # Load water texture (texture unit 1) - Keep as standard 2D texture for now
        # Actually water uses 'water' shader which is distinct.
        self.water_texture = self.create_texture('water_texture.png', has_alpha=True)
        if self.water_texture:
            self.water_texture.use(1)
            print("Water texture loaded on unit 1")
        
        return self.block_texture is not None
    
    def bind_texture(self, texture, slot=0):
        """Bind a texture to a texture slot"""
        if texture:
            texture.use(slot)
            if self.chunk_program:
                self.chunk_program['u_texture_0'] = slot
    
    def bind_water_texture(self):
        """Bind water texture for water surface rendering"""
        if self.water_texture:
            self.water_texture.use(1)  # Use texture unit 1 for water
            water_program = self.shader_manager.get_program('water')
            if water_program:
                water_program['u_texture_0'] = 1
    
    def render_sky(self, view_matrix, time):
        """Render the sky background"""
        if hasattr(self, 'sky_renderer'):
            self.sky_renderer.render(view_matrix, self.proj_matrix, time)
            
    def render_water_surface(self, view_matrix, camera_pos):
        """Render the water surface plane"""
        if self.water_surface:
            self.water_surface.render(view_matrix, self.proj_matrix, camera_pos)
            
    def toggle_wireframe(self):
        """Toggle wireframe mode"""
        if hasattr(self, 'wireframe_mode') and self.wireframe_mode:
            self.ctx.wireframe = False
            self.wireframe_mode = False
            print("Wireframe mode: OFF")
        else:
            self.ctx.wireframe = True
            self.wireframe_mode = True
            print("Wireframe mode: ON")

    # ------------------------------------------------------------------
    # Crosshair
    # ------------------------------------------------------------------

    def _init_crosshair(self):
        """Create the 2-D crosshair shader program and geometry."""
        vert_src = """
#version 330 core
in vec2 in_pos;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""
        frag_src = """
#version 330 core
out vec4 fragColor;
void main() {
    fragColor = vec4(1.0, 1.0, 1.0, 1.0);
}
"""
        self.crosshair_prog = self.ctx.program(
            vertex_shader=vert_src,
            fragment_shader=frag_src,
        )
        self._build_crosshair_vao()

    def _build_crosshair_vao(self):
        """Build crosshair geometry in NDC space."""
        # Two lines: horizontal and vertical
        # Each occupies ~2% of the half-screen width/height
        s = 0.018   # arm half-length in NDC
        t = 0.002   # pixel gap in centre (tiny gap like MC's crosshair)
        verts = np.array([
            # horizontal
            -s,  0.0,
            -t,  0.0,
             t,  0.0,
             s,  0.0,
            # vertical
             0.0, -s,
             0.0, -t,
             0.0,  t,
             0.0,  s,
        ], dtype=np.float32)
        vbo = self.ctx.buffer(verts.tobytes())
        self.crosshair_vao = self.ctx.vertex_array(
            self.crosshair_prog,
            [(vbo, '2f', 'in_pos')],
        )

    def render_crosshair(self):
        """Draw a small white crosshair at the centre of the screen."""
        # Disable depth test so it always appears on top
        self.ctx.disable(mgl.DEPTH_TEST)
        self.ctx.disable(mgl.CULL_FACE)

        # Make lines a bit thicker so they are clearly visible
        self.ctx.line_width = 2.0

        self.crosshair_vao.render(mgl.LINES)

        # Restore state
        self.ctx.enable(mgl.DEPTH_TEST)
        self.ctx.enable(mgl.CULL_FACE)
        self.ctx.line_width = 1.0

    # ------------------------------------------------------------------
    # Block outline
    # ------------------------------------------------------------------

    def _init_block_outline(self):
        """Create the 3-D block-outline shader program and geometry."""
        vert_src = """
#version 330 core
in vec3 in_pos;
uniform mat4 m_proj;
uniform mat4 m_view;
uniform mat4 m_model;
void main() {
    gl_Position = m_proj * m_view * m_model * vec4(in_pos, 1.0);
}
"""
        frag_src = """
#version 330 core
out vec4 fragColor;
void main() {
    fragColor = vec4(0.0, 0.0, 0.0, 1.0);
}
"""
        self.outline_prog = self.ctx.program(
            vertex_shader=vert_src,
            fragment_shader=frag_src,
        )

        # Unit cube, slightly expanded by eps so it sits just outside the block
        eps = 0.005
        lo = 0.0 - eps
        hi = 1.0 + eps
        # 12 edges  x  2 vertices = 24 vertices
        edges = [
            # bottom face
            lo, lo, lo,  hi, lo, lo,
            hi, lo, lo,  hi, lo, hi,
            hi, lo, hi,  lo, lo, hi,
            lo, lo, hi,  lo, lo, lo,
            # top face
            lo, hi, lo,  hi, hi, lo,
            hi, hi, lo,  hi, hi, hi,
            hi, hi, hi,  lo, hi, hi,
            lo, hi, hi,  lo, hi, lo,
            # vertical edges
            lo, lo, lo,  lo, hi, lo,
            hi, lo, lo,  hi, hi, lo,
            hi, lo, hi,  hi, hi, hi,
            lo, lo, hi,  lo, hi, hi,
        ]
        verts = np.array(edges, dtype=np.float32)
        vbo = self.ctx.buffer(verts.tobytes())
        self.outline_vao = self.ctx.vertex_array(
            self.outline_prog,
            [(vbo, '3f', 'in_pos')],
        )

    def render_block_outline(self, block_pos, view_matrix, proj_matrix):
        """Draw a black wireframe outline around the block at *block_pos*.

        *block_pos* is a tuple/list of three integers (world coordinates).
        """
        x, y, z = block_pos
        model = glm.translate(glm.mat4(1.0), glm.vec3(x - 0.005, y - 0.005, z - 0.005))

        self.outline_prog['m_proj'].write(proj_matrix.to_bytes())
        self.outline_prog['m_view'].write(view_matrix.to_bytes())
        self.outline_prog['m_model'].write(model.to_bytes())

        # Draw on top of the block without z-fighting
        self.ctx.disable(mgl.CULL_FACE)
        self.ctx.enable(mgl.DEPTH_TEST)
        self.ctx.line_width = 2.0

        # Slightly pull the lines toward the camera to avoid z-fighting
        self.ctx.enable(mgl.BLEND)
        self.ctx.blend_func = mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA

        self.outline_vao.render(mgl.LINES)

        # Restore
        self.ctx.enable(mgl.CULL_FACE)
        self.ctx.line_width = 1.0
