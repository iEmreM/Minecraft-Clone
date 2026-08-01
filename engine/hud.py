"""
HUD Renderer — hotbar and the creative block picker, drawn with ModernGL
(OpenGL 3.3 core). Works correctly with pg.OPENGL | pg.DOUBLEBUF display mode.

Both surfaces share one pair of programs: flat-coloured quads for backgrounds
and borders, textured quads for block icons out of the same texture array the
world is drawn with. The block list itself comes from world/blocks.py — this
file knows how to lay icons out, not which blocks exist.
"""

import math

import numpy as np
import moderngl as mgl

from world.blocks import CREATIVE, HOTBAR_DEFAULT, ICON_LAYER

HOTBAR_SLOTS = len(HOTBAR_DEFAULT)

# Colour and atlas layer ride along on the vertices rather than sitting in a
# uniform. That is the whole reason the hotbar is two draw calls instead of 54:
# with the colour in a uniform, every quad needed its own draw. The creative
# grid is 123 icons and gets the same treatment for the same reason.
_VERT_COLOR = """
#version 330 core
in vec2 in_pos;
in vec4 in_color;
out vec4 v_color;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    v_color = in_color;
}
"""

_FRAG_COLOR = """
#version 330 core
in vec4 v_color;
out vec4 fragColor;
void main() { fragColor = v_color; }
"""

_VERT_TEX = """
#version 330 core
in vec2 in_pos;
in vec2 in_uv;
in float in_layer;
out vec2 v_uv;
flat out float v_layer;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    v_uv = in_uv;
    v_layer = in_layer;
}
"""

_FRAG_TEX = """
#version 330 core
in vec2 v_uv;
flat in float v_layer;
uniform sampler2DArray u_tex;
out vec4 fragColor;
void main() {
    fragColor = texture(u_tex, vec3(v_uv, v_layer));
}
"""


def _ndc(px, total):
    """Convert pixel coordinate to NDC [-1, 1] (y-up)."""
    return (px / total) * 2.0 - 1.0


def _quad_verts(x0, y0, x1, y1, u0=0., v0=1., u1=1., v1=0.):
    """Return 6 vertices (2 triangles) as a flat list for a rectangle in NDC.

    v runs 1 at the bottom edge to 0 at the top: NDC y is up, but the atlas
    arrives from pg.image.tostring in top-to-bottom row order, so v=0 is the
    top of the tile. Matching them the other way drew every icon upside down —
    invisible while the atlas held ten near-symmetric textures, obvious the
    moment it held a furnace and a TNT crate.
    """
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
    """Hotbar plus the creative block picker."""

    SLOT_PX   = 52
    GAP_PX    = 4
    MARGIN_PX = 8   # icon inset inside slot
    BOTTOM_PY = 14  # pixels from bottom of screen

    # Creative grid. The column count is fixed and the cell size is fitted to
    # the window, so every block is on screen at once and there is nothing to
    # scroll — 123 blocks in 12 columns is 11 rows, which fits any window the
    # game is playable in.
    INV_COLS = 12

    def __init__(self, ctx: mgl.Context, screen_w: int, screen_h: int):
        self.ctx = ctx
        self.screen_w = screen_w
        self.screen_h = screen_h

        self.color_prog = ctx.program(
            vertex_shader=_VERT_COLOR, fragment_shader=_FRAG_COLOR)
        self.tex_prog = ctx.program(
            vertex_shader=_VERT_TEX, fragment_shader=_FRAG_TEX)

        # Hotbar contents are mutable now: the creative window writes into them.
        self.hotbar = list(HOTBAR_DEFAULT)
        self.inventory_open = False

        # One buffer per program, both written only when the hotbar's contents
        # change — which is on a resize, on the selected slot moving and on a
        # slot being reassigned, not once a frame.
        self._color_vbo = None
        self._color_vao = None
        self._icon_vbo = None
        self._icon_vao = None
        self._color_slot = None      # slot the colour buffer currently holds
        self._inv_color_vbo = None
        self._inv_color_vao = None
        self._inv_icon_vbo = None
        self._inv_icon_vao = None
        self._inv_hover = -2         # hover index the inventory colours hold
        self._build_geometry()
        self._build_inventory()

    # ------------------------------------------------------------------
    def resize(self, screen_w: int, screen_h: int):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self._build_geometry()
        self._build_inventory()

    def set_slot(self, slot: int, block_id: int):
        """Put *block_id* in a hotbar slot — what clicking a creative icon does."""
        if self.hotbar[slot] == block_id:
            return
        self.hotbar[slot] = block_id
        self._build_geometry()

    # ------------------------------------------------------------------
    def _build_geometry(self):
        """Pre-compute slot rectangles in NDC space."""
        sw, sh = self.screen_w, self.screen_h
        n         = HOTBAR_SLOTS
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
            })

        icon_verts = []
        for i, block_id in enumerate(self.hotbar):
            layer = ICON_LAYER.get(block_id)
            if layer is None:
                continue
            flat = _quad_verts(*self._slots[i]['icon'])
            for v in range(6):
                icon_verts.extend(flat[v * 4:v * 4 + 4])
                icon_verts.append(float(layer))

        data = np.array(icon_verts, dtype=np.float32)
        self._icon_count = len(icon_verts) // 5
        if self._icon_vao is not None:
            self._icon_vao.release()
            self._icon_vbo.release()
        self._icon_vbo = self.ctx.buffer(data.tobytes())
        self._icon_vao = self.ctx.vertex_array(
            self.tex_prog, [(self._icon_vbo, '2f 2f 1f', 'in_pos', 'in_uv', 'in_layer')])

        # Backgrounds and borders share one buffer; it is rewritten only when
        # the highlighted slot moves.
        if self._color_vao is not None:
            self._color_vao.release()
            self._color_vbo.release()
        self._color_vbo = self.ctx.buffer(
            reserve=HOTBAR_SLOTS * 5 * 6 * 6 * 4, dynamic=True)
        self._color_vao = self.ctx.vertex_array(
            self.color_prog, [(self._color_vbo, '2f 4f', 'in_pos', 'in_color')])
        self._color_slot = None
        self._color_count = 0

    # ------------------------------------------------------------------
    @staticmethod
    def _push_quad(out, x0, y0, x1, y1, color):
        for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y0), (x1, y1), (x0, y1)):
            out.append(x)
            out.append(y)
            out.extend(color)

    def _build_color_data(self, selected_slot: int):
        """Background and border quads for every slot, in the order they used to
        be drawn one at a time."""
        out = []
        for i in range(HOTBAR_SLOTS):
            s = self._slots[i]
            x0, y0, x1, y1 = s['bg']

            if i == selected_slot:
                self._push_quad(out, x0, y0, x1, y1, (1.0, 1.0, 1.0, 0.25))
                bw = abs(_ndc(2, self.screen_w) - _ndc(0, self.screen_w))  # 2px in NDC
                bh = abs(_ndc(2, self.screen_h) - _ndc(0, self.screen_h))
                color = (1.0, 0.85, 0.0, 1.0)   # golden yellow
            else:
                self._push_quad(out, x0, y0, x1, y1, (0.1, 0.1, 0.1, 0.55))
                bw = abs(_ndc(1, self.screen_w) - _ndc(0, self.screen_w))
                bh = abs(_ndc(1, self.screen_h) - _ndc(0, self.screen_h))
                color = (0.6, 0.6, 0.6, 0.9)

            self._push_quad(out, x0, y0, x1, y0 + bh, color)   # bottom
            self._push_quad(out, x0, y1 - bh, x1, y1, color)   # top
            self._push_quad(out, x0, y0, x0 + bw, y1, color)   # left
            self._push_quad(out, x1 - bw, y0, x1, y1, color)   # right

        return np.array(out, dtype=np.float32)

    # ------------------------------------------------------------------
    # Creative block picker
    # ------------------------------------------------------------------

    def _build_inventory(self):
        """Lay the whole block list out in a grid and build its icon buffer.

        Cell size is fitted to the window rather than fixed, which is what lets
        the grid stay scroll-free: shrinking the window shrinks the icons
        instead of hiding blocks behind a scrollbar nobody can see without a
        font to label it.
        """
        sw, sh = self.screen_w, self.screen_h
        cols = self.INV_COLS
        rows = math.ceil(len(CREATIVE) / cols)

        # The panel lives above the hotbar, so the hotbar stays visible and
        # usable while the picker is open — clicking an icon fills the selected
        # slot, and you can watch it land.
        top_of_hotbar = self.BOTTOM_PY + self.SLOT_PX + 12
        avail_h = max(sh - top_of_hotbar - 24, 40)

        cell = int(min(46, sw * 0.82 / cols, avail_h * 0.94 / rows))
        cell = max(cell, 10)
        gap = max(2, cell // 14)
        pad = max(6, cell // 2)

        grid_w = cols * cell + (cols - 1) * gap
        grid_h = rows * cell + (rows - 1) * gap
        px0 = (sw - grid_w) / 2
        # Centre the panel in the space above the hotbar.
        py0 = top_of_hotbar + (avail_h - grid_h) / 2

        def ndcx(px): return (px / sw) * 2.0 - 1.0
        def ndcy(py): return (py / sh) * 2.0 - 1.0

        self._inv_cells = []      # (block_id, px rect y-up, ndc rect)
        icon_verts = []
        for i, block_id in enumerate(CREATIVE):
            cx = px0 + (i % cols) * (cell + gap)
            cy = py0 + grid_h - cell - (i // cols) * (cell + gap)
            rect_px = (cx, cy, cx + cell, cy + cell)
            rect_ndc = (ndcx(cx), ndcy(cy), ndcx(cx + cell), ndcy(cy + cell))
            self._inv_cells.append((block_id, rect_px, rect_ndc))

            inset = max(2, cell // 10)
            flat = _quad_verts(ndcx(cx + inset), ndcy(cy + inset),
                               ndcx(cx + cell - inset), ndcy(cy + cell - inset))
            layer = float(ICON_LAYER[block_id])
            for v in range(6):
                icon_verts.extend(flat[v * 4:v * 4 + 4])
                icon_verts.append(layer)

        self._inv_panel = (ndcx(px0 - pad), ndcy(py0 - pad),
                           ndcx(px0 + grid_w + pad), ndcy(py0 + grid_h + pad))

        data = np.array(icon_verts, dtype=np.float32)
        self._inv_icon_count = len(icon_verts) // 5
        if self._inv_icon_vao is not None:
            self._inv_icon_vao.release()
            self._inv_icon_vbo.release()
        self._inv_icon_vbo = self.ctx.buffer(data.tobytes())
        self._inv_icon_vao = self.ctx.vertex_array(
            self.tex_prog,
            [(self._inv_icon_vbo, '2f 2f 1f', 'in_pos', 'in_uv', 'in_layer')])

        if self._inv_color_vao is not None:
            self._inv_color_vao.release()
            self._inv_color_vbo.release()
        # Screen dim + panel + one background per cell + a 4-quad hover border.
        quads = 2 + len(CREATIVE) + 4
        self._inv_color_vbo = self.ctx.buffer(reserve=quads * 6 * 6 * 4, dynamic=True)
        self._inv_color_vao = self.ctx.vertex_array(
            self.color_prog, [(self._inv_color_vbo, '2f 4f', 'in_pos', 'in_color')])
        self._inv_hover = -2
        self._inv_color_count = 0

    def _build_inventory_color_data(self, hover: int):
        out = []
        self._push_quad(out, -1.0, -1.0, 1.0, 1.0, (0.0, 0.0, 0.0, 0.45))
        self._push_quad(out, *self._inv_panel, (0.08, 0.08, 0.10, 0.92))
        for i, (_, _, rect) in enumerate(self._inv_cells):
            shade = (0.35, 0.35, 0.38, 0.95) if i == hover else (0.22, 0.22, 0.25, 0.9)
            self._push_quad(out, *rect, shade)

        if 0 <= hover < len(self._inv_cells):
            x0, y0, x1, y1 = self._inv_cells[hover][2]
            bw = abs(_ndc(2, self.screen_w) - _ndc(0, self.screen_w))
            bh = abs(_ndc(2, self.screen_h) - _ndc(0, self.screen_h))
            color = (1.0, 0.85, 0.0, 1.0)
            self._push_quad(out, x0, y0, x1, y0 + bh, color)
            self._push_quad(out, x0, y1 - bh, x1, y1, color)
            self._push_quad(out, x0, y0, x0 + bw, y1, color)
            self._push_quad(out, x1 - bw, y0, x1, y1, color)

        return np.array(out, dtype=np.float32)

    def hit_test(self, mouse_x: int, mouse_y: int):
        """Grid index under the mouse, or -1.

        *mouse_y* is pygame's, measured from the top; the layout is y-up.
        """
        py = self.screen_h - mouse_y
        for i, (_, (x0, y0, x1, y1), _) in enumerate(self._inv_cells):
            if x0 <= mouse_x <= x1 and y0 <= py <= y1:
                return i
        return -1

    def block_at(self, index: int):
        """Block id for a grid index from hit_test, or None."""
        if 0 <= index < len(self._inv_cells):
            return self._inv_cells[index][0]
        return None

    def render_inventory(self, hover: int, block_texture_array):
        """Draw the creative picker over the frame. Call after render()."""
        ctx = self.ctx
        ctx.disable(mgl.DEPTH_TEST)
        ctx.disable(mgl.CULL_FACE)
        ctx.enable(mgl.BLEND)
        ctx.blend_func = mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA

        if self._inv_hover != hover:
            data = self._build_inventory_color_data(hover)
            self._inv_color_vbo.write(data.tobytes())
            self._inv_color_count = len(data) // 6
            self._inv_hover = hover

        self._inv_color_vao.render(mgl.TRIANGLES, vertices=self._inv_color_count)

        if block_texture_array:
            block_texture_array.use(0)
            self.tex_prog['u_tex'].value = 0
            self._inv_icon_vao.render(mgl.TRIANGLES, vertices=self._inv_icon_count)

        ctx.enable(mgl.DEPTH_TEST)
        ctx.enable(mgl.CULL_FACE)

    # ------------------------------------------------------------------
    def render(self, selected_slot: int, block_texture_array):
        """
        Call this after all 3-D rendering, before pg.display.flip().

        Two draw calls: every background and border in one, every icon in the
        other. Backgrounds still land under the icons, and the borders sit in
        the 8-pixel margin outside them, so the result is what the 54-call
        version drew.

        :param selected_slot:      Active hotbar slot index (0-based).
        :param block_texture_array: The ModernGL Texture3D / TextureArray bound to unit 0.
        """
        ctx = self.ctx
        ctx.disable(mgl.DEPTH_TEST)
        ctx.disable(mgl.CULL_FACE)
        ctx.enable(mgl.BLEND)
        ctx.blend_func = mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA

        if self._color_slot != selected_slot:
            data = self._build_color_data(selected_slot)
            self._color_vbo.write(data.tobytes())
            self._color_count = len(data) // 6
            self._color_slot = selected_slot

        self._color_vao.render(mgl.TRIANGLES, vertices=self._color_count)

        if block_texture_array:
            block_texture_array.use(0)
            self.tex_prog['u_tex'].value = 0
            self._icon_vao.render(mgl.TRIANGLES, vertices=self._icon_count)

        # Restore state
        ctx.enable(mgl.DEPTH_TEST)
        ctx.enable(mgl.CULL_FACE)
