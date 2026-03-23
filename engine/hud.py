"""
HUD Renderer — draws the hotbar using ModernGL (OpenGL 3.3 core).
Works correctly with pg.OPENGL | pg.DOUBLEBUF display mode.
"""

import numpy as np
import moderngl as mgl

# Hotbar definition — WATER removed intentionally
HOTBAR_BLOCKS = [1, 2, 3, 4, 5, 6, 7, 9, 10]   # block-type IDs, slot 0-8

BLOCK_NAMES = {
    1: "Grass",
    2: "Dirt",
    3: "Stone",
    4: "Sand",
    5: "Snow",
    6: "Leaves",
    7: "Wood",
    9: "Stone Brick",
    10: "Brick",
}

# Atlas texture-array layer: layer = col + row * 4
# Corrected coordinates (verified by user):
BLOCK_ATLAS_LAYER = {
    1:  1 + 3 * 4,   # GRASS top        = layer 13
    2:  0 + 2 * 4,   # DIRT             = layer  8
    3:  0 + 1 * 4,   # STONE            = layer  4
    4:  1 + 2 * 4,   # SAND             = layer  9
    5:  3 + 3 * 4,   # SNOW             = layer 15
    6:  1 + 0 * 4,   # LEAVES           = layer  1
    7:  2 + 1 * 4,   # WOOD side        = layer  6
    9:  2 + 2 * 4,   # STONE_BRICK      = layer 10
    10: 2 + 3 * 4,   # BRICK            = layer 14
}

_VERT_COLOR = """
#version 330 core
in vec2 in_pos;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

_FRAG_COLOR = """
#version 330 core
uniform vec4 u_color;
out vec4 fragColor;
void main() { fragColor = u_color; }
"""

_VERT_TEX = """
#version 330 core
in vec2 in_pos;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    v_uv = in_uv;
}
"""

_FRAG_TEX = """
#version 330 core
in vec2 v_uv;
uniform sampler2DArray u_tex;
uniform float u_layer;
out vec4 fragColor;
void main() {
    fragColor = texture(u_tex, vec3(v_uv, u_layer));
}
"""


def _ndc(px, total):
    """Convert pixel coordinate to NDC [-1, 1] (y-up)."""
    return (px / total) * 2.0 - 1.0


def _quad_verts(x0, y0, x1, y1, u0=0., v0=0., u1=1., v1=1.):
    """Return 6 vertices (2 triangles) as a flat list for a rectangle in NDC."""
    # Each vertex: x, y, u, v
    return [
        x0, y0, u0, v0,
        x1, y0, u1, v0,
        x1, y1, u1, v1,
        x0, y0, u0, v0,
        x1, y1, u1, v1,
        x0, y1, u0, v1,
    ]


class HUDRenderer:
    """Renders the 9-slot hotbar using ModernGL directly."""

    SLOT_PX   = 52
    GAP_PX    = 4
    MARGIN_PX = 8   # icon inset inside slot
    BOTTOM_PY = 14  # pixels from bottom of screen

    def __init__(self, ctx: mgl.Context, screen_w: int, screen_h: int):
        self.ctx = ctx
        self.screen_w = screen_w
        self.screen_h = screen_h

        self.color_prog = ctx.program(
            vertex_shader=_VERT_COLOR, fragment_shader=_FRAG_COLOR)
        self.tex_prog = ctx.program(
            vertex_shader=_VERT_TEX, fragment_shader=_FRAG_TEX)

        # Placeholder VBOs — rebuilt in _build_geometry
        self._dummy = np.array([0.0] * 48, dtype=np.float32)
        self._slot_color_vbo = ctx.buffer(self._dummy.tobytes(), dynamic=True)
        self._slot_color_vao = ctx.vertex_array(
            self.color_prog, [(self._slot_color_vbo, '2f', 'in_pos')])

        self._icon_vbo = ctx.buffer(self._dummy.tobytes(), dynamic=True)
        self._icon_vao = ctx.vertex_array(
            self.tex_prog, [(self._icon_vbo, '2f 2f', 'in_pos', 'in_uv')])

        self._geom_cache = {}  # slot_index -> (bg_verts, icon_verts, border_verts)
        self._build_geometry()

    # ------------------------------------------------------------------
    def resize(self, screen_w: int, screen_h: int):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self._geom_cache.clear()
        self._build_geometry()

    def get_selected_block(self, slot: int) -> int:
        return HOTBAR_BLOCKS[slot % len(HOTBAR_BLOCKS)]

    # ------------------------------------------------------------------
    def _build_geometry(self):
        """Pre-compute slot rectangles in NDC space."""
        sw, sh = self.screen_w, self.screen_h
        n         = len(HOTBAR_BLOCKS)
        total_px  = n * self.SLOT_PX + (n - 1) * self.GAP_PX
        x0_base   = (sw - total_px) / 2          # pixel x of first slot left edge
        y0_base   = self.BOTTOM_PY               # pixel y from bottom (y-up)

        self._slots = []
        for i in range(n):
            # Slot outer rect in pixels (y measured from bottom)
            sx0 = x0_base + i * (self.SLOT_PX + self.GAP_PX)
            sy0 = y0_base
            sx1 = sx0 + self.SLOT_PX
            sy1 = sy0 + self.SLOT_PX

            # Icon inner rect (shrink by MARGIN_PX on all sides)
            m = self.MARGIN_PX
            ix0, iy0, ix1, iy1 = sx0 + m, sy0 + m, sx1 - m, sy1 - m

            # Convert pixels → NDC (y-up: 0 = bottom of screen)
            def px2ndcx(px): return (px / sw) * 2.0 - 1.0
            def px2ndcy(py): return (py / sh) * 2.0 - 1.0

            self._slots.append({
                'bg':   (px2ndcx(sx0), px2ndcy(sy0), px2ndcx(sx1), px2ndcy(sy1)),
                'icon': (px2ndcx(ix0), px2ndcy(iy0), px2ndcx(ix1), px2ndcy(iy1)),
                # border thicknesses in NDC units
                'bw': (2.0 / sw) * 2.0,
                'bh': (2.0 / sh) * 2.0,
            })


    # ------------------------------------------------------------------
    def _draw_colored_quad(self, x0, y0, x1, y1, r, g, b, a):
        verts = np.array([
            x0, y0,  x1, y0,  x1, y1,
            x0, y0,  x1, y1,  x0, y1,
        ], dtype=np.float32)
        self._slot_color_vbo.orphan(len(verts) * 4)
        self._slot_color_vbo.write(verts.tobytes())
        self.color_prog['u_color'].value = (r, g, b, a)
        self._slot_color_vao.render(mgl.TRIANGLES, vertices=6)

    def _draw_textured_quad(self, x0, y0, x1, y1, layer: float):
        verts = np.array(_quad_verts(x0, y0, x1, y1), dtype=np.float32)
        self._icon_vbo.orphan(len(verts) * 4)
        self._icon_vbo.write(verts.tobytes())
        self.tex_prog['u_layer'].value = layer
        self._icon_vao.render(mgl.TRIANGLES, vertices=6)

    # ------------------------------------------------------------------
    def render(self, selected_slot: int, block_texture_array):
        """
        Call this after all 3-D rendering, before pg.display.flip().

        :param selected_slot:      Active hotbar slot index (0-based).
        :param block_texture_array: The ModernGL Texture3D / TextureArray bound to unit 0.
        """
        ctx = self.ctx
        ctx.disable(mgl.DEPTH_TEST)
        ctx.disable(mgl.CULL_FACE)
        ctx.enable(mgl.BLEND)
        ctx.blend_func = mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA

        # Bind the block texture array for icon drawing
        if block_texture_array:
            block_texture_array.use(0)
            self.tex_prog['u_tex'].value = 0

        for i, block_id in enumerate(HOTBAR_BLOCKS):
            s = self._slots[i]
            x0, y0, x1, y1 = s['bg']

            # ---- slot background ----
            if i == selected_slot:
                self._draw_colored_quad(x0, y0, x1, y1, 1.0, 1.0, 1.0, 0.25)
            else:
                self._draw_colored_quad(x0, y0, x1, y1, 0.1, 0.1, 0.1, 0.55)

            # ---- block icon ----
            if block_id in BLOCK_ATLAS_LAYER:
                ix0, iy0, ix1, iy1 = s['icon']
                self._draw_textured_quad(ix0, iy0, ix1, iy1,
                                         float(BLOCK_ATLAS_LAYER[block_id]))

            # ---- slot border ----
            if i == selected_slot:
                # Draw 4 thin colored lines as quads (top / bottom / left / right)
                bw = abs(_ndc(2, self.screen_w) - _ndc(0, self.screen_w))  # 2px in NDC
                bh = abs(_ndc(2, self.screen_h) - _ndc(0, self.screen_h))
                color = (1.0, 0.85, 0.0, 1.0)   # golden yellow
                # bottom
                self._draw_colored_quad(x0, y0, x1, y0 + bh, *color)
                # top
                self._draw_colored_quad(x0, y1 - bh, x1, y1, *color)
                # left
                self._draw_colored_quad(x0, y0, x0 + bw, y1, *color)
                # right
                self._draw_colored_quad(x1 - bw, y0, x1, y1, *color)
            else:
                bw = abs(_ndc(1, self.screen_w) - _ndc(0, self.screen_w))
                bh = abs(_ndc(1, self.screen_h) - _ndc(0, self.screen_h))
                color = (0.6, 0.6, 0.6, 0.9)
                self._draw_colored_quad(x0, y0, x1, y0 + bh, *color)
                self._draw_colored_quad(x0, y1 - bh, x1, y1, *color)
                self._draw_colored_quad(x0, y0, x0 + bw, y1, *color)
                self._draw_colored_quad(x1 - bw, y0, x1, y1, *color)

        # Restore state
        ctx.enable(mgl.DEPTH_TEST)
        ctx.enable(mgl.CULL_FACE)
