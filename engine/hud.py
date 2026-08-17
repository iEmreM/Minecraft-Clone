"""
HUD Renderer — hotbar and the creative block picker, drawn with ModernGL
(OpenGL 3.3 core). Works correctly with pg.OPENGL | pg.DOUBLEBUF display mode.

Everything shares three programs: flat-coloured quads for backgrounds and
borders, `sampler2DArray` quads for block icons out of the same texture array
the world is drawn with, and `sampler2D` quads for text. The block list itself
comes from world/blocks.py — this file knows how to lay icons out, not which
blocks exist.
"""

import math

import numpy as np
import moderngl as mgl
import pygame as pg

from world.blocks import (BLOCK_NAMES, CREATIVE, GROUPS, HOTBAR_DEFAULT,
                          ICON_LAYER)

HOTBAR_SLOTS = len(HOTBAR_DEFAULT)

# The picker's tab row: one page per category, plus an "everything" page that
# search results also land on.
ALL_TAB = 'Tümü'
TABS = [(ALL_TAB, CREATIVE)] + [(name, list(ids)) for name, ids in GROUPS]

# Colour and atlas layer ride along on the vertices rather than sitting in a
# uniform. That is the whole reason the hotbar is two draw calls instead of 54:
# with the colour in a uniform, every quad needed its own draw. The creative
# grid is 254 icons and gets the same treatment for the same reason.
#
# `u_offset` is how the block list scrolls: the list is laid out once at scroll
# zero and the whole thing is shifted at draw time, so scrolling costs one
# uniform write rather than a rebuild of every vertex.
_VERT_COLOR = """
#version 330 core
in vec2 in_pos;
in vec4 in_color;
uniform vec2 u_offset;
out vec4 v_color;
void main() {
    gl_Position = vec4(in_pos + u_offset, 0.0, 1.0);
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
uniform vec2 u_offset;
out vec2 v_uv;
flat out float v_layer;
void main() {
    gl_Position = vec4(in_pos + u_offset, 0.0, 1.0);
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

# Text is a plain 2-D texture. Runs of labels that share a font — the tab row,
# the group titles — are baked into one strip each, so a row of tabs is one
# draw rather than one per tab.
_VERT_TEXT = """
#version 330 core
in vec2 in_pos;
in vec2 in_uv;
uniform vec2 u_offset;
out vec2 v_uv;
void main() {
    gl_Position = vec4(in_pos + u_offset, 0.0, 1.0);
    v_uv = in_uv;
}
"""

_FRAG_TEXT = """
#version 330 core
in vec2 v_uv;
uniform sampler2D u_tex;
out vec4 fragColor;
void main() { fragColor = texture(u_tex, v_uv); }
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

    # The picker is a fixed fraction of the window, not a box fitted to its
    # contents: it used to resize on every keystroke as the result count moved,
    # which reads as the window jumping around under the cursor. Fixed frame,
    # scrolling contents.
    PANEL_W_R = 0.62
    PANEL_H_R = 0.72
    CELL_R    = 0.055   # target icon size as a fraction of window height

    def __init__(self, ctx: mgl.Context, screen_w: int, screen_h: int):
        self.ctx = ctx
        self.screen_w = screen_w
        self.screen_h = screen_h

        self.color_prog = ctx.program(
            vertex_shader=_VERT_COLOR, fragment_shader=_FRAG_COLOR)
        self.tex_prog = ctx.program(
            vertex_shader=_VERT_TEX, fragment_shader=_FRAG_TEX)
        self.text_prog = ctx.program(
            vertex_shader=_VERT_TEXT, fragment_shader=_FRAG_TEXT)

        # The block texture array is shared with the terrain pass, which needs
        # LINEAR magnification for the shader-side nearest snap that stops
        # distant blocks crawling. An icon has no such derivatives — it is a
        # 16px tile blown up 3x on a flat quad — so LINEAR there is only blur.
        # A sampler object overrides the texture's own filter for as long as it
        # is bound, which is the one way to disagree without mutating shared
        # state and having to remember to put it back.
        self._icon_sampler = ctx.sampler(filter=(mgl.NEAREST, mgl.NEAREST))

        pg.font.init()

        # Hotbar contents are mutable now: the creative window writes into them.
        self.hotbar = list(HOTBAR_DEFAULT)
        self.inventory_open = False
        self.query = ''            # search box contents
        self.tab = 0               # index into TABS
        self.scroll = 0.0          # pixels the block list is scrolled down by

        # One buffer per program, both written only when the hotbar's contents
        # change — which is on a resize, on the selected slot moving and on a
        # slot being reassigned, not once a frame.
        self._color_vbo = None
        self._color_vao = None
        self._icon_vbo = None
        self._icon_vao = None
        self._color_slot = None      # slot the colour buffer currently holds
        self._cell_vbo = None
        self._cell_vao = None
        self._chrome_vbo = None
        self._chrome_vao = None
        self._chrome_key = None      # (tab, scroll, hover-in-tabs) chrome holds
        self._inv_icon_vbo = None
        self._inv_icon_vao = None
        self._inv_hover = -2         # hover index the cell colours hold
        self._header = _TextStrip()
        self._tabs_text = _TextStrip()
        self._label_tex = None       # tooltip under the cursor
        self._tip_rect = None
        self._query_tex = None       # what the search box shows
        self._query_shown = None
        self._debug_left = None      # the F3 screen, one texture per column
        self._debug_right = None
        self._debug_shown = None
        self._mouse_px = (0, 0)
        self._fonts = {}
        self._text_vbo = ctx.buffer(reserve=6 * 4 * 4, dynamic=True)
        self._text_vao = ctx.vertex_array(
            self.text_prog, [(self._text_vbo, '2f 2f', 'in_pos', 'in_uv')])
        self._tip_vbo = ctx.buffer(reserve=6 * 6 * 4, dynamic=True)
        self._tip_vao = ctx.vertex_array(
            self.color_prog, [(self._tip_vbo, '2f 4f', 'in_pos', 'in_color')])
        self._build_geometry()
        self._build_inventory()

    # ------------------------------------------------------------------
    def resize(self, screen_w: int, screen_h: int):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self._build_geometry()
        self._build_inventory()
        self._debug_shown = None    # same text, new font size and new corners

    def set_query(self, query: str):
        """Type into the search box. Empty string shows the current tab again."""
        if query == self.query:
            return
        # A search spans every category, so it shows on the page that holds
        # every category — otherwise typing "diamond" on the wool tab finds
        # nothing and looks broken.
        if query and not self.query:
            self.tab = 0
        self.query = query
        self.scroll = 0.0
        self._build_inventory()

    def set_tab(self, index: int):
        """Switch category page. Clears the search, which spans all of them."""
        if not 0 <= index < len(TABS) or index == self.tab:
            return
        self.tab = index
        self.query = ''
        self.scroll = 0.0
        self._build_inventory()

    def scroll_by(self, pixels: float):
        self.set_scroll(self.scroll + pixels)

    def set_scroll(self, value: float):
        self.scroll = min(max(value, 0.0), self._max_scroll)

    @property
    def scroll_step(self):
        """One wheel notch: a row of icons."""
        return self._cell + self._gap

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
                'px':   (sx0, sy0, sx1, sy1),
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
        _release(self._icon_vao, self._icon_vbo)
        self._icon_vbo = self.ctx.buffer(data.tobytes())
        self._icon_vao = self.ctx.vertex_array(
            self.tex_prog, [(self._icon_vbo, '2f 2f 1f', 'in_pos', 'in_uv', 'in_layer')])

        # Backgrounds and borders share one buffer; it is rewritten only when
        # the highlighted slot moves.
        _release(self._color_vao, self._color_vbo)
        self._color_vbo = self.ctx.buffer(
            reserve=HOTBAR_SLOTS * 5 * 6 * 6 * 4, dynamic=True)
        self._color_vao = self.ctx.vertex_array(
            self.color_prog, [(self._color_vbo, '2f 4f', 'in_pos', 'in_color')])
        self._color_slot = None
        self._color_count = 0

    def hotbar_slot_at(self, mouse_x: int, mouse_y: int):
        """Hotbar slot under the mouse, or -1.

        The hotbar stays live while the picker is open — with the wheel taken
        over by the block list, clicking a slot is how you choose where the next
        pick lands.
        """
        py = self.screen_h - mouse_y
        for i, slot in enumerate(self._slots):
            x0, y0, x1, y1 = slot['px']
            if x0 <= mouse_x <= x1 and y0 <= py <= y1:
                return i
        return -1

    # ------------------------------------------------------------------
    @staticmethod
    def _push_quad(out, x0, y0, x1, y1, color):
        for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y0), (x1, y1), (x0, y1)):
            out.append(x)
            out.append(y)
            out.extend(color)

    def _push_border(self, out, x0, y0, x1, y1, color, px=2):
        bw = abs(_ndc(px, self.screen_w) - _ndc(0, self.screen_w))
        bh = abs(_ndc(px, self.screen_h) - _ndc(0, self.screen_h))
        self._push_quad(out, x0, y0, x1, y0 + bh, color)   # bottom
        self._push_quad(out, x0, y1 - bh, x1, y1, color)   # top
        self._push_quad(out, x0, y0, x0 + bw, y1, color)   # left
        self._push_quad(out, x1 - bw, y0, x1, y1, color)   # right

    def _build_color_data(self, selected_slot: int):
        """Background and border quads for every slot, in the order they used to
        be drawn one at a time."""
        out = []
        for i in range(HOTBAR_SLOTS):
            s = self._slots[i]
            x0, y0, x1, y1 = s['bg']

            if i == selected_slot:
                self._push_quad(out, x0, y0, x1, y1, (1.0, 1.0, 1.0, 0.25))
                self._push_border(out, x0, y0, x1, y1, (1.0, 0.85, 0.0, 1.0), 2)
            else:
                self._push_quad(out, x0, y0, x1, y1, (0.1, 0.1, 0.1, 0.55))
                self._push_border(out, x0, y0, x1, y1, (0.6, 0.6, 0.6, 0.9), 1)

        return np.array(out, dtype=np.float32)

    # ------------------------------------------------------------------
    # Text
    # ------------------------------------------------------------------

    def _font(self, size):
        """pygame's built-in font at *size*, cached — the search box rebuilds
        its text on every keystroke and Font() is not cheap."""
        font = self._fonts.get(size)
        if font is None:
            font = self._fonts[size] = pg.font.Font(None, size)
        return font

    def _text_texture(self, surface):
        """A pygame Surface as a GL texture, rows bottom-up so v runs with NDC y."""
        tex = self.ctx.texture(surface.get_size(), 4,
                               pg.image.tostring(surface, 'RGBA', True))
        tex.filter = (mgl.LINEAR, mgl.LINEAR)
        return tex

    def _build_strip(self, strip, items, font):
        """Bake `(text, colour, px x, px y)` labels into one texture and one VAO.

        A texture per label would be a draw call per label; the tab row alone is
        twelve of them, and they only change when the layout does.
        """
        strip.release()
        if not items:
            return
        rendered = [font.render(text, True, color) for text, color, _, _ in items]
        strip_w = max(s.get_width() for s in rendered)
        strip_h = sum(s.get_height() for s in rendered)
        surface = pg.Surface((strip_w, strip_h), pg.SRCALPHA)

        verts = []
        top = 0
        for surf, (_, _, x, y) in zip(rendered, items):
            surface.blit(surf, (0, top))
            w, h = surf.get_size()
            # The strip is uploaded bottom-up, so a row `top` from the top of the
            # surface sits `strip_h - top - h` from the bottom.
            verts += _quad_verts(
                _ndc(x, self.screen_w), _ndc(y, self.screen_h),
                _ndc(x + w, self.screen_w), _ndc(y + h, self.screen_h),
                0.0, (strip_h - top - h) / strip_h,
                w / strip_w, (strip_h - top) / strip_h)
            top += h

        strip.tex = self._text_texture(surface)
        strip.vbo = self.ctx.buffer(np.array(verts, dtype=np.float32).tobytes())
        strip.vao = self.ctx.vertex_array(
            self.text_prog, [(strip.vbo, '2f 2f', 'in_pos', 'in_uv')])
        strip.count = len(verts) // 4

    def _draw_text(self, tex, x, y, w, h):
        """Blit a text texture at a pixel rect (y measured from the bottom)."""
        verts = _quad_verts(
            _ndc(x, self.screen_w), _ndc(y, self.screen_h),
            _ndc(x + w, self.screen_w), _ndc(y + h, self.screen_h),
            0.0, 0.0, 1.0, 1.0)
        self._text_vbo.write(np.array(verts, dtype=np.float32).tobytes())
        tex.use(0)
        self.text_prog['u_tex'].value = 0
        self._text_vao.render(mgl.TRIANGLES, vertices=6)

    def _set_offset(self, dy_px):
        """Shift everything drawn next by *dy_px* — how the block list scrolls."""
        value = (0.0, 2.0 * dy_px / self.screen_h)
        for prog in (self.color_prog, self.tex_prog, self.text_prog):
            prog['u_offset'].value = value

    # ------------------------------------------------------------------
    # Creative block picker
    # ------------------------------------------------------------------

    def _visible_groups(self):
        """(title, block ids) sections the picker should list right now.

        A search spans every group and says which one each hit came from; a
        plain category page is one section and needs no title, because the tab
        above it already says the same word.
        """
        q = self.query.strip().lower()
        if q:
            return [(name, hits) for name, ids in GROUPS
                    for hits in ([b for b in ids if q in BLOCK_NAMES[b].lower()],)
                    if hits]
        name, ids = TABS[self.tab]
        if name != ALL_TAB:
            return [(None, ids)]
        return [(g, list(i)) for g, i in GROUPS]

    def _build_inventory(self):
        """Lay the panel out: tab row, search box, scrolling block list.

        Runs on a resize, a tab switch and every keystroke. The panel frame is
        fixed — only what goes inside it changes.
        """
        sw, sh = self.screen_w, self.screen_h

        # The panel lives above the hotbar, so the hotbar stays visible and
        # usable while the picker is open — clicking an icon fills the selected
        # slot, and you can watch it land.
        top_of_hotbar = self.BOTTOM_PY + self.SLOT_PX + 12
        avail_h = max(sh - top_of_hotbar - 20, 80)
        panel_w = max(sw * self.PANEL_W_R, 240.0)
        panel_h = min(sh * self.PANEL_H_R, avail_h)
        px0 = (sw - panel_w) / 2
        py0 = top_of_hotbar + (avail_h - panel_h) / 2
        self._panel_px = (px0, py0, px0 + panel_w, py0 + panel_h)
        self._panel = (_ndc(px0, sw), _ndc(py0, sh),
                       _ndc(px0 + panel_w, sw), _ndc(py0 + panel_h, sh))

        pad = max(8.0, panel_w * 0.018)
        inner_x0 = px0 + pad
        inner_w = panel_w - 2 * pad
        self._font_px = max(13, int(sh * 0.022))

        tabs_h = self._layout_tabs(inner_x0, py0 + panel_h - pad, inner_w)

        search_h = self._font_px * 1.9
        search_top = py0 + panel_h - pad - tabs_h - pad * 0.6
        self._search_rect = (inner_x0, search_top - search_h,
                             inner_x0 + inner_w, search_top)

        # Viewport: everything under the search box. The scrollbar track sits
        # inside it on the right, so the icons stop short of it.
        vp_top = search_top - search_h - pad * 0.6
        vp_bottom = py0 + pad
        self._sb_w = max(6.0, pad * 0.7)
        self._viewport = (inner_x0, vp_bottom, inner_x0 + inner_w, vp_top)
        content_w = inner_w - self._sb_w - 6.0

        cell_target = max(24.0, sh * self.CELL_R)
        self._gap = gap = max(2.0, cell_target * 0.12)
        cols = max(4, int((content_w + gap) / (cell_target + gap)))
        self._cell = cell = (content_w - (cols - 1) * gap) / cols

        self._build_list(self._visible_groups(), inner_x0, vp_top, cols, cell, gap)
        self.set_scroll(self.scroll)
        self._chrome_key = None
        self._inv_hover = -2
        self._query_shown = None

    def _layout_tabs(self, x0, top, width):
        """Wrap the tab labels into rows and bake them. Returns the row height."""
        font = self._font(self._font_px)
        pad_x, pad_y = self._font_px * 0.45, self._font_px * 0.30
        sizes = [font.size(name) for name, _ in TABS]
        row_h = max(h for _, h in sizes) + 2 * pad_y

        self._tabs = []           # (px rect y-up) per tab, index-aligned to TABS
        labels = []
        x, y = x0, top - row_h
        for i, (name, _) in enumerate(TABS):
            w = sizes[i][0] + 2 * pad_x
            if x > x0 and x + w > x0 + width:
                x, y = x0, y - row_h - 3.0
            self._tabs.append((x, y, x + w, y + row_h))
            labels.append((name, (255, 255, 255) if i == self.tab else (168, 168, 176),
                           x + pad_x, y + pad_y))
            x += w + 3.0

        self._build_strip(self._tabs_text, labels, font)
        return top - y

    def _build_list(self, sections, x0, top, cols, cell, gap):
        """Cells, icons and section titles for the visible blocks.

        Laid out at scroll zero; `u_offset` moves it at draw time.
        """
        sw, sh = self.screen_w, self.screen_h
        title_h = self._font_px * 1.35
        inset = max(2.0, cell * 0.1)

        self._inv_cells = []      # (block_id, px rect y-up at scroll 0, ndc rect)
        icon_verts = []
        titles = []
        y = top
        for name, ids in sections:
            if name is not None:
                titles.append((name, (255, 214, 92), x0, y - title_h + self._font_px * 0.2))
                y -= title_h
            for i, block_id in enumerate(ids):
                cx = x0 + (i % cols) * (cell + gap)
                cy = y - cell - (i // cols) * (cell + gap)
                self._inv_cells.append((
                    block_id, (cx, cy, cx + cell, cy + cell),
                    (_ndc(cx, sw), _ndc(cy, sh),
                     _ndc(cx + cell, sw), _ndc(cy + cell, sh))))

                flat = _quad_verts(
                    _ndc(cx + inset, sw), _ndc(cy + inset, sh),
                    _ndc(cx + cell - inset, sw), _ndc(cy + cell - inset, sh))
                layer = float(ICON_LAYER[block_id])
                for v in range(6):
                    icon_verts.extend(flat[v * 4:v * 4 + 4])
                    icon_verts.append(layer)
            y -= math.ceil(len(ids) / cols) * (cell + gap)

        self._content_h = top - y
        self._max_scroll = max(0.0, self._content_h
                               - (self._viewport[3] - self._viewport[1]))

        self._build_strip(self._header, titles, self._font(int(self._font_px * 0.95)))

        data = np.array(icon_verts, dtype=np.float32)
        self._inv_icon_count = len(icon_verts) // 5
        _release(self._inv_icon_vao, self._inv_icon_vbo)
        self._inv_icon_vbo = self.ctx.buffer(data.tobytes() or b'\0' * 4)
        self._inv_icon_vao = self.ctx.vertex_array(
            self.tex_prog,
            [(self._inv_icon_vbo, '2f 2f 1f', 'in_pos', 'in_uv', 'in_layer')])

        _release(self._cell_vao, self._cell_vbo)
        self._cell_vbo = self.ctx.buffer(
            reserve=(len(self._inv_cells) + 4) * 6 * 6 * 4, dynamic=True)
        self._cell_vao = self.ctx.vertex_array(
            self.color_prog, [(self._cell_vbo, '2f 4f', 'in_pos', 'in_color')])
        self._cell_count = 0

        _release(self._chrome_vao, self._chrome_vbo)
        # Dim, panel, search box, scrollbar track and thumb, one background and
        # a 4-quad border per tab.
        self._chrome_vbo = self.ctx.buffer(
            reserve=(5 + len(TABS) * 5) * 6 * 6 * 4, dynamic=True)
        self._chrome_vao = self.ctx.vertex_array(
            self.color_prog, [(self._chrome_vbo, '2f 4f', 'in_pos', 'in_color')])
        self._chrome_count = 0

    def _build_chrome(self, tab_hover):
        """Panel frame: dim, background, tabs, search box, scrollbar."""
        sw, sh = self.screen_w, self.screen_h

        def rect(px, color):
            self._push_quad(out, _ndc(px[0], sw), _ndc(px[1], sh),
                            _ndc(px[2], sw), _ndc(px[3], sh), color)

        out = []
        self._push_quad(out, -1.0, -1.0, 1.0, 1.0, (0.0, 0.0, 0.0, 0.5))
        rect(self._panel_px, (0.08, 0.08, 0.10, 0.94))

        for i, box in enumerate(self._tabs):
            if i == self.tab:
                rect(box, (0.30, 0.30, 0.34, 1.0))
                self._push_border(out, _ndc(box[0], sw), _ndc(box[1], sh),
                                  _ndc(box[2], sw), _ndc(box[3], sh),
                                  (1.0, 0.85, 0.0, 1.0), 2)
            else:
                rect(box, (0.19, 0.19, 0.22, 1.0) if i == tab_hover
                     else (0.14, 0.14, 0.16, 1.0))

        rect(self._search_rect, (0.02, 0.02, 0.03, 0.96))

        if self._max_scroll > 0:
            vx0, vy0, vx1, vy1 = self._viewport
            track = (vx1 - self._sb_w, vy0, vx1, vy1)
            rect(track, (0.16, 0.16, 0.19, 0.95))
            view_h = vy1 - vy0
            thumb_h = max(view_h * view_h / self._content_h, 18.0)
            travel = view_h - thumb_h
            t = self.scroll / self._max_scroll
            top = vy1 - t * travel
            rect((track[0], top - thumb_h, track[2], top), (0.55, 0.55, 0.60, 1.0))

        return np.array(out, dtype=np.float32)

    def _build_cell_data(self, hover: int):
        """Cell backgrounds and the hover border, in list space (scrolled)."""
        out = []
        for i, (_, _, rect) in enumerate(self._inv_cells):
            shade = (0.35, 0.35, 0.38, 0.95) if i == hover else (0.22, 0.22, 0.25, 0.9)
            self._push_quad(out, *rect, shade)
        if 0 <= hover < len(self._inv_cells):
            self._push_border(out, *self._inv_cells[hover][2],
                              color=(1.0, 0.85, 0.0, 1.0), px=2)
        return np.array(out, dtype=np.float32)

    def _set_hover(self, hover):
        """Build the tooltip for the cell under the cursor.

        The name is the whole point of the tooltip, so it is rendered here
        rather than at draw time: the background quad has to be sized to the
        text, and it is drawn after the icons so it cannot ride along in a
        buffer that goes out before them.
        """
        _release(self._label_tex)
        self._label_tex = None
        self._tip_rect = None
        block_id = self.block_at(hover)
        if block_id is None:
            return

        surf = self._font(self._font_px + 3).render(
            BLOCK_NAMES[block_id], True, (245, 245, 245))
        w, h = surf.get_size()
        mx, my = self._mouse_px
        pad = 6
        x = mx + 16
        if x + w + 2 * pad > self.screen_w:
            x = max(mx - 16 - w - 2 * pad, 0)
        y = max(min(my - h, self.screen_h - h - 2 * pad), pad)
        self._label_tex = self._text_texture(surf)
        self._tip_rect = (x, y, w, h, pad)

        bg = []
        self._push_quad(bg, _ndc(x, self.screen_w), _ndc(y - pad, self.screen_h),
                        _ndc(x + w + 2 * pad, self.screen_w),
                        _ndc(y + h + pad, self.screen_h), (0.04, 0.04, 0.06, 0.94))
        self._tip_vbo.write(np.array(bg, dtype=np.float32).tobytes())

    # ------------------------------------------------------------------
    def hit_test(self, mouse_x: int, mouse_y: int):
        """Grid index under the mouse, or -1.

        *mouse_y* is pygame's, measured from the top; the layout is y-up. Cells
        are stored unscrolled, so the cursor is moved into list space and cells
        outside the viewport are not hittable however far they have scrolled.
        """
        py = self.screen_h - mouse_y
        self._mouse_px = (mouse_x, py)
        vx0, vy0, vx1, vy1 = self._viewport
        if not (vx0 <= mouse_x <= vx1 - self._sb_w and vy0 <= py <= vy1):
            return -1
        ly = py - self.scroll
        for i, (_, (x0, y0, x1, y1), _) in enumerate(self._inv_cells):
            if x0 <= mouse_x <= x1 and y0 <= ly <= y1:
                return i
        return -1

    def tab_at(self, mouse_x: int, mouse_y: int):
        """Tab index under the mouse, or -1."""
        py = self.screen_h - mouse_y
        for i, (x0, y0, x1, y1) in enumerate(self._tabs):
            if x0 <= mouse_x <= x1 and y0 <= py <= y1:
                return i
        return -1

    def scrollbar_at(self, mouse_x: int, mouse_y: int):
        """True if the mouse is on the scrollbar track."""
        vx0, vy0, vx1, vy1 = self._viewport
        py = self.screen_h - mouse_y
        return (self._max_scroll > 0 and vy0 <= py <= vy1
                and vx1 - self._sb_w <= mouse_x <= vx1)

    def scroll_to_mouse(self, mouse_y: int):
        """Jump the list so the thumb centres on the cursor — click or drag."""
        _, vy0, _, vy1 = self._viewport
        py = self.screen_h - mouse_y
        self.set_scroll((1.0 - (py - vy0) / max(vy1 - vy0, 1.0)) * self._max_scroll)

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

        mx, py = self._mouse_px          # stored y-up; tab_at wants pygame's
        tab_hover = self.tab_at(mx, self.screen_h - py)
        key = (self.tab, round(self.scroll), tab_hover)
        if self._chrome_key != key:
            data = self._build_chrome(tab_hover)
            self._chrome_vbo.write(data.tobytes())
            self._chrome_count = len(data) // 6
            self._chrome_key = key

        if self._inv_hover != hover:
            self._set_hover(hover)
            data = self._build_cell_data(hover)
            self._cell_vbo.write(data.tobytes())
            self._cell_count = len(data) // 6
            self._inv_hover = hover

        self._set_offset(0.0)
        self._chrome_vao.render(mgl.TRIANGLES, vertices=self._chrome_count)

        # The list is taller than the panel; the scissor is what keeps it inside.
        vx0, vy0, vx1, vy1 = self._viewport
        ctx.scissor = (int(vx0), int(vy0), int(vx1 - vx0), int(vy1 - vy0))
        self._set_offset(self.scroll)
        if self._cell_count:
            self._cell_vao.render(mgl.TRIANGLES, vertices=self._cell_count)
        if block_texture_array and self._inv_icon_count:
            block_texture_array.use(0)
            self._icon_sampler.use(0)
            self.tex_prog['u_tex'].value = 0
            self._inv_icon_vao.render(mgl.TRIANGLES, vertices=self._inv_icon_count)
            self._icon_sampler.clear(0)
        self._header.draw(self.text_prog)
        ctx.scissor = None

        self._set_offset(0.0)
        self._tabs_text.draw(self.text_prog)
        self._render_search()
        if self._label_tex is not None:
            self._tip_vao.render(mgl.TRIANGLES, vertices=6)
            x, y, w, h, pad = self._tip_rect
            self._draw_text(self._label_tex, x + pad, y, w, h)

        ctx.enable(mgl.DEPTH_TEST)
        ctx.enable(mgl.CULL_FACE)

    def _render_search(self):
        """The search box: what has been typed, or the hint, or 'no results'."""
        if self.query:
            text, color = self.query + '|', (255, 255, 255)
            if not self._inv_cells:
                text, color = self.query + '|   (sonuç yok)', (255, 150, 150)
        else:
            text, color = 'Ara...', (140, 140, 148)

        if self._query_shown != text:
            _release(self._query_tex)
            self._query_tex = self._text_texture(
                self._font(self._font_px).render(text, True, color))
            self._query_shown = text

        sx0, sy0, sx1, sy1 = self._search_rect
        w, h = self._query_tex.size
        self._draw_text(self._query_tex, sx0 + 8, sy0 + (sy1 - sy0 - h) / 2, w, h)

    # ------------------------------------------------------------------
    # Debug screen (F3)
    # ------------------------------------------------------------------

    DEBUG_BG = (0, 0, 0, 165)      # backdrop behind each line, as the real one has
    DEBUG_FG = (232, 232, 232)

    def set_debug(self, left, right):
        """Put text on the F3 screen: one tuple of lines per column.

        Tuples, because an unchanged screen has to cost one comparison — and the
        text really does change ten times a second, which is why a column is one
        texture and one quad rather than a `_TextStrip`. A strip would rebuild a
        buffer and a VAO at that rate to place lines that a single blit has
        already placed.
        """
        if (left, right) == self._debug_shown:
            return
        self._debug_shown = (left, right)
        _release(self._debug_left, self._debug_right)
        self._debug_left = self._bake_column(left)
        self._debug_right = self._bake_column(right, right_align=True)

    def _bake_column(self, lines, right_align=False):
        """A column of debug lines, each on its own backdrop, as one texture.

        An empty line is a gap — no backdrop, no text — so a column reads as
        groups rather than as one slab of numbers.
        """
        font = self._font(max(18, int(self.screen_h * 0.033)))
        pad, step = 3, font.get_linesize()
        rendered = [font.render(text, True, self.DEBUG_FG) for text in lines]
        width = max((s.get_width() for s in rendered), default=0) + 2 * pad
        column = pg.Surface((max(width, 1), max(step * len(lines), 1)), pg.SRCALPHA)

        for i, surf in enumerate(rendered):
            if not lines[i]:
                continue
            x = width - pad - surf.get_width() if right_align else pad
            column.fill(self.DEBUG_BG,
                        (x - pad, i * step, surf.get_width() + 2 * pad, step))
            column.blit(surf, (x, i * step))
        return self._text_texture(column)

    def render_debug(self):
        """Draw the two columns in the top corners. Call after render()."""
        if self._debug_left is None:
            return

        ctx = self.ctx
        ctx.disable(mgl.DEPTH_TEST)
        ctx.disable(mgl.CULL_FACE)
        ctx.enable(mgl.BLEND)
        ctx.blend_func = mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA
        self._set_offset(0.0)

        for tex, right in ((self._debug_left, False), (self._debug_right, True)):
            w, h = tex.size
            self._draw_text(tex, self.screen_w - w if right else 0,
                            self.screen_h - h, w, h)

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
        self._set_offset(0.0)

        if self._color_slot != selected_slot:
            data = self._build_color_data(selected_slot)
            self._color_vbo.write(data.tobytes())
            self._color_count = len(data) // 6
            self._color_slot = selected_slot

        self._color_vao.render(mgl.TRIANGLES, vertices=self._color_count)

        if block_texture_array:
            block_texture_array.use(0)
            self._icon_sampler.use(0)
            self.tex_prog['u_tex'].value = 0
            self._icon_vao.render(mgl.TRIANGLES, vertices=self._icon_count)
            self._icon_sampler.clear(0)

        # Restore state
        ctx.enable(mgl.DEPTH_TEST)
        ctx.enable(mgl.CULL_FACE)


class _TextStrip:
    """One texture holding several labels, plus the quads that index into it."""

    def __init__(self):
        self.tex = self.vbo = self.vao = None
        self.count = 0

    def release(self):
        _release(self.tex, self.vao, self.vbo)
        self.tex = self.vbo = self.vao = None
        self.count = 0

    def draw(self, prog):
        if self.vao is None:
            return
        self.tex.use(0)
        prog['u_tex'].value = 0
        self.vao.render(mgl.TRIANGLES, vertices=self.count)


def _release(*objs):
    for obj in objs:
        if obj is not None:
            obj.release()
