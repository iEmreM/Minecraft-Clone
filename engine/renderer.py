import moderngl as mgl
import pygame as pg
import numpy as np
from engine.shader_manager import ShaderManager
from engine.water_surface import WaterSurface
from engine.sky import SkyRenderer
from world.terrain_generator import WATER_LINE
from world.blocks import LAYER_COUNT
import glm
import math


class ModernGLRenderer:
    FOV_DEGREES = 65.0
    NEAR_PLANE = 0.1
    FAR_PLANE = 1000.0

    # The sky gradient, and the only copy of it. sky.frag draws it, and chunk /
    # water fog resolve to the same gradient along the view ray, so distant
    # terrain dissolves into the sky instead of standing out against it.
    BG_COLOR = (0.6, 0.8, 0.95)      # horizon
    SKY_ZENITH = (0.0, 0.4, 0.8)

    # Fog starts this far into the render distance and is opaque at the end of it.
    FOG_START_FRACTION = 0.65
    FOG_END_FRACTION = 0.9           # of render_distance * CHUNK_SIZE

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
        
        # Get chunk shader programs. The two draw the same vertex format with
        # the same uniforms and differ only in their fragment shader, so every
        # uniform write below fans out over `chunk_programs` — a uniform written
        # to one and not the other is how the glass ends up fogged differently
        # from the wall around it.
        self.chunk_program = self.shader_manager.get_program('chunk')
        self.chunk_alpha_program = self.shader_manager.get_program('chunk_alpha')
        self.chunk_programs = tuple(p for p in (self.chunk_program,
                                                self.chunk_alpha_program) if p)
        
        # Store screen dimensions
        self.width = width
        self.height = height
        self.wireframe_mode = False

        # Initialize textures
        self.block_texture = None
        self.water_texture = None
        
        # Initialize water surface
        self.water_surface = WaterSurface(self)
        
        # Initialize sky renderer
        self.sky_renderer = SkyRenderer(self.ctx, self.shader_manager)
        
        self.bg_color = glm.vec3(*self.BG_COLOR)
        self.sky_zenith = glm.vec3(*self.SKY_ZENITH)
        self.proj_matrix = self._make_projection(width, height)
        self._upload_static_uniforms()
        self.set_fog_distance(96.0)  # replaced by the chunk manager's render distance

        # Initialize crosshair and block outline
        self._init_crosshair()
        self._init_block_outline()
        
        print("ModernGL Renderer initialized successfully")
    
    def _make_projection(self, width, height):
        """Perspective matrix for the current viewport.

        One place, so startup and resize cannot disagree: the far plane used to
        be 1000 in __init__ and 5000 in resize, so the first window resize
        silently changed the render range and cost depth precision. Fog is fully
        opaque well before 1000 anyway.
        """
        return glm.perspective(glm.radians(self.FOV_DEGREES), width / height,
                               self.NEAR_PLANE, self.FAR_PLANE)

    def _upload_static_uniforms(self):
        """Write the chunk uniforms that only change on a resize.

        The sky gradient, the water line and the projection are all fixed between
        resizes; they used to be re-uploaded on every frame, and the water line
        came with a module import each time on top of that.
        """
        # The sky gradient goes to every program that has to agree on it.
        for name in ('chunk', 'chunk_alpha', 'water', 'sky'):
            program = self.shader_manager.get_program(name)
            if program and 'sky_horizon' in program:
                program['sky_horizon'].write(self.bg_color)
                program['sky_zenith'].write(self.sky_zenith)

        if self.water_surface:
            self.water_surface.upload_static_uniforms()

        for program in self.chunk_programs:
            program['m_proj'].write(self.proj_matrix.to_bytes())
            program['water_line'] = float(WATER_LINE)

    def set_fog_distance(self, view_radius):
        """Fit the fog to how far the world actually loads.

        `view_radius` is render_distance * CHUNK_SIZE, straight from the chunk
        manager. Fog is opaque at FOG_END_FRACTION of it — before the load edge,
        so terrain never pops in — and starts at FOG_START_FRACTION of that, so
        everything nearer stays clear. Water reads the same range and sizes its
        plane from it, so land and sea cannot disagree about where the world ends.
        """
        self.fog_end = max(float(view_radius), 1.0) * self.FOG_END_FRACTION
        self.fog_start = self.fog_end * self.FOG_START_FRACTION
        fog_range = glm.vec2(self.fog_start, self.fog_end)

        for program in self.chunk_programs:
            program['fog_range'].write(fog_range)
        if self.water_surface:
            self.water_surface.set_fog(fog_range, self.fog_end)

    def resize(self, width, height):
        """Handle window resize"""
        self.width = width
        self.height = height
        self.ctx.viewport = (0, 0, width, height)
        self.proj_matrix = self._make_projection(width, height)
        self._upload_static_uniforms()

    def clear(self):
        """Clear the screen with sky background color like ornek2"""
        self.ctx.clear(color=self.BG_COLOR)
    
    def set_view_matrix(self, view_matrix):
        """Set the view matrix for rendering"""
        if self.chunk_program:
            self.chunk_program['m_view'].write(view_matrix.to_bytes())
    
    def update_matrices(self, view_matrix, model_matrix=None):
        """Upload the two chunk uniforms that change every frame.

        Everything else lives in _upload_static_uniforms.
        """
        # OPTIMIZATION: m_model removed - shader compiler optimizes it out since unused
        # The vertex shader doesn't use m_model (always identity), so GLSL compiler removes it
        # Eye position for radial fog. Taken from the view matrix rather than
        # added to this method's signature, so no caller can forget to pass it.
        self.cam_pos = glm.vec3(glm.inverse(view_matrix)[3])
        view_bytes = view_matrix.to_bytes()
        for program in self.chunk_programs:
            program['m_view'].write(view_bytes)
            program['cam_pos'].write(self.cam_pos)
    
    def create_vao(self, vertices, indices=None, transparent=False):
        """Create a Vertex Array Object from vertex data.

        Returns (vao, vbo, ibo). The caller owns all three and must release all
        three — releasing the VAO alone leaves its buffers on the GPU.

        *transparent* picks the see-through program. The vertex format is
        identical; a VAO is bound to one program, so the chunk's two meshes need
        one each.
        """
        if vertices.size == 0:
            return None, None, None

        # copy=False so the already-correct dtype coming out of the mesher is
        # uploaded as is; astype used to copy ~130 KB per chunk for nothing.
        vbo = self.ctx.buffer(vertices.astype(np.float32, copy=False).tobytes())

        ibo = (self.ctx.buffer(indices.astype(np.uint32, copy=False).tobytes())
               if indices is not None else None)

        # Updated format: 3f position, 3f tex_coord (vec3), 1f shading
        vao = self.ctx.vertex_array(
            self.chunk_alpha_program if transparent else self.chunk_program,
            [(vbo, '3f 3f 1f', 'in_position', 'in_tex_coord', 'in_shading')],
            ibo)

        return vao, vbo, ibo

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
            
            texture_array.repeat_x = True # Allow tiling
            texture_array.repeat_y = True
            texture_array.build_mipmaps()

            # Order matters: build_mipmaps() overwrites filter with
            # (LINEAR_MIPMAP_LINEAR, LINEAR), so setting it first silently
            # threw the choice away and every block was bilinearly upscaled.
            # That was invisible while a tile was 64x64 — LINEAR magnification
            # only engages once a block covers more than one screen pixel per
            # texel, i.e. inside ~10 blocks — but a 16x16 tile crosses that
            # line at ~39 blocks, so the whole near field went soft.
            #
            # Magnification is LINEAR because the *shader* does the nearest
            # snap now (`sample_nearest` in chunk_common.glsl, the reference's
            # terrain.fsh). A NEAREST sampler put the boundary between two
            # texels wholly inside one pixel, so walking made it jump a pixel
            # at a time across every repeating surface at once — the crawling
            # the blocky look was costing us. The shader lands on the same
            # texel centres and uses this bilinear tap only to resolve the last
            # screen pixel of the edge, so blocks stay hard and stop shimmering.
            # The HUD binds its own NEAREST sampler over this texture: an icon
            # is magnified with no derivatives to speak of and there LINEAR is
            # only blur.
            #
            # Anisotropy is what matters at distance: a chunk seen at a grazing
            # angle is minified far harder along one axis than the other, and an
            # isotropic sampler has to pick the blurrier of the two for both,
            # which is what turns distant ground into mush. The driver clamps
            # the 16 to whatever it supports.
            texture_array.filter = (mgl.LINEAR_MIPMAP_LINEAR, mgl.LINEAR)
            texture_array.anisotropy = 16.0


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
        # texture.png is a single 16-wide column of 16x16 tiles, baked by
        # build_atlas.py — so a tile's row *is* its array layer, and
        # world/blocks.py can hand out layer numbers without knowing the atlas
        # shape. It used to be a 4x4 grid, which capped the game at 16 textures.
        self.block_texture = self.create_texture_array('texture.png', 1, LAYER_COUNT)
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
            for program in self.chunk_programs:
                program['u_texture_0'] = slot
    
    def bind_water_texture(self):
        """Bind water texture for water surface rendering"""
        if self.water_texture:
            self.water_texture.use(1)  # Use texture unit 1 for water
            water_program = self.shader_manager.get_program('water')
            if water_program:
                water_program['u_texture_0'] = 1
    
    def render_sky(self, view_matrix, time):
        """Render the sky background, and give the fog the same clock.

        chunk/water fog resolves to sky_color(), whose clouds drift with u_time.
        Feeding all three from this one call is what stops the clouds inside the
        fog from sliding away from the ones drawn in the sky.
        """
        for name in ('chunk', 'chunk_alpha', 'water'):
            program = self.shader_manager.get_program(name)
            if program and 'u_time' in program:
                program['u_time'] = time

        if hasattr(self, 'sky_renderer'):
            self.sky_renderer.render(view_matrix, self.proj_matrix, time)
            
    def render_water_surface(self, view_matrix, camera_pos):
        """Render the water surface plane"""
        if self.water_surface:
            self.water_surface.render(view_matrix, self.proj_matrix, camera_pos)
            
    def toggle_wireframe(self):
        """Toggle wireframe mode. The flag is what the debug screen reads."""
        self.wireframe_mode = self.ctx.wireframe = not self.wireframe_mode
        print(f"Wireframe mode: {'ON' if self.wireframe_mode else 'OFF'}")

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
