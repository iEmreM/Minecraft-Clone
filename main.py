import pygame as pg
import sys
import time
from engine.renderer import ModernGLRenderer
from engine.camera import Camera
from world.modern_chunk import AIR, GRASS, WATER
from world.blocks import BLOCK_NAMES, COLLIDES, FACING, LOWER, UPPER, WALL_MOUNTED
from world.threaded_chunk_manager import ThreadedChunkManager
from world.terrain_generator import BIOME_NAMES, column_biome, find_spawn
from world.shapes import FACING_NAMES
import glm
import math
import commands
from engine.hud import HUDRenderer, HOTBAR_SLOTS

try:                                   # the debug screen's memory and CPU lines
    import psutil
    _PROC, _RAM = psutil.Process(), psutil.virtual_memory
    _RAM_MB = _RAM().total >> 20
except ImportError:                    # not a game dependency; the lines just go
    _PROC = None

# What the crosshair goes straight through, and what a placed block overwrites.
# Water is in it for the same reason it is in the real game: the ray does not
# stop on it, so the sea is never outlined and never breakable, and a block
# aimed at the sea bed lands *in* the water instead of stacking on its surface.
# One tuple, because the outline and the placement have to agree — a ray that
# passes through a block it then refuses to build in targets nothing at all.
REPLACEABLE = (AIR, WATER)

# Which world axis each compass name points along, for the debug screen.
LOOK_AXIS = {'north': '-Z', 'east': '+X', 'south': '+Z', 'west': '-X'}

# What opens the command console. `/` is a *key*, and on a layout where the
# slash is not a bare key press — a Turkish Q keyboard, for one — that key press
# never arrives; T always does, and the slash can be typed inside. The numpad
# divide is there because it is the same character on every layout.
CONSOLE_KEYS = (pg.K_t, pg.K_SLASH, pg.K_KP_DIVIDE)


def look_direction(yaw):
    """Compass name and world axis the eye points along, from a camera yaw.

    `Camera.yaw` is 0 down +X and turns toward +Z, while `shapes.FACING_NAMES`
    starts at -Z — hence the quarter turn. Off by one and the screen still reads
    plausibly while pointing the player the wrong way, so `test_interaction.py`
    checks the axis against the front vector the camera itself builds.
    """
    name = FACING_NAMES[(int(yaw % 360 + 45) // 90 + 1) % 4]
    return name, LOOK_AXIS[name]


class MinecraftModernGL:
    def __init__(self, width=800, height=600):
        # Initialize renderer
        self.renderer = ModernGLRenderer(width, height)
        
        # Initialize camera on the first dry column out from the origin. A fixed
        # spawn height puts the player inside a mountain or on a sea bed as soon
        # as the terrain has either.
        self.camera = Camera(position=find_spawn())
        
        # Game state
        self.running = True
        self.clock = pg.time.Clock()
        self.delta_time = 0
        self.last_frame = time.time()
        
        # Mouse control
        self.mouse_captured = False
        self.last_x = width / 2
        self.last_y = height / 2
        self.first_mouse = True
        self.start_time = time.time()

        # Debug screen (F3). The render stats it shows are written by render(),
        # so they start at zero rather than being missing on the first frame.
        self.show_debug = False
        self.last_debug_update = 0.0
        self.last_cpu_sample = 0.0
        self.cpu_percent = 0.0
        self.target = {'hit': False}
        self.rendered_chunks = self.total_chunks = 0
        self.frustum_culled = self.triangles = 0
        info = self.renderer.ctx.info      # ~60 GL queries; once is enough
        self.gpu = (info['GL_RENDERER'], info['GL_VERSION'])


        # Movement
        self.movement_speed = 20.0
        self.mouse_sensitivity = 0.3
        
        # Block interaction
        self.block_reach = 8.0  # How far the player can reach
        self.selected_block_type = GRASS  # GRASS by default
        self.hotbar_slot = 0              # Active hotbar slot (0-based)
        
        # World state with threaded chunk manager
        self.render_distance = 6  # Configurable render distance
        self.chunk_manager = ThreadedChunkManager(self.renderer, self.render_distance)
        self.texture = None
        
        # Set world reference for camera collision detection
        self.camera.set_world(self.chunk_manager)
        
        self.initialize_spawn_chunks()
        
        # Load texture
        self.load_texture()
        
        # HUD / hotbar (must be created after renderer/ctx is ready)
        self.hud = HUDRenderer(self.renderer.ctx, self.renderer.width, self.renderer.height)
        self.inv_hover = -1        # creative grid index under the mouse
        self._was_captured = True  # mouse state to restore when a window closes
        self._swallow_text = False # drop the TEXTINPUT of the key that opened it
        self.dragging_scrollbar = False

        # Command console (T, or / for a line with the slash already in it).
        # The picker and the console both take the whole keyboard, so they can
        # never be open at once and share _was_captured / _swallow_text.
        self.console_open = False
        self.console_text = ''
        self.console_log = []      # (text, colour, time it was printed)
        self.console_history = []  # submitted lines, oldest first
        self.console_at = 0        # where UP/DOWN has walked back to in it
        self.console_draft = ''    # the line UP was pressed from, to come back to

        print("Minecraft ModernGL initialized successfully!")
        print("Controls:")
        print("- WASD: Move (walk/fly)")
        print("- Mouse: Look around")
        print("- Space: Jump (walking) / Up (flying)")
        print("- Shift: Down (flying only)")
        print("- TAB: Toggle flying mode")
        print("- ESC: Toggle mouse capture (closes the block picker)")
        print("- Click to capture mouse")
        print("- Left Click: Remove block")
        print("- Right Click: Place block")
        print(f"- E: Creative block picker — click a block to put it in the selected slot")
        print("     category tabs at the top, type to search, wheel scrolls the list,")
        print("     click a hotbar slot to change where the next pick lands")
        print(f"- 1-{HOTBAR_SLOTS}: Select hotbar slot")
        print("- +/-: Increase/Decrease render distance")
        print("- F: Toggle frustum culling")
        print("- F3: Debug screen — position, chunk, biome, target, FPS, memory")
        print("- T or /: Command console — /help lists the commands,")
        print("     /tp teleports, /locate finds a biome, /set changes a setting")
        print("")
        print("Physics: Gravity, jumping, and block collision enabled!")
        print("Walking speed: 5 blocks/sec, Flying speed: 15 blocks/sec")
    
    def initialize_spawn_chunks(self):
        """Initialize chunks around spawn position with pre-generation"""
        spawn_pos = self.camera.position
        
        # Pre-generate initial chunks synchronously
        print("Pre-generating initial chunks around spawn...")
        self.chunk_manager.pregenerate_spawn_chunks(spawn_pos.x, spawn_pos.z)
        
        # Get current chunk for additional loading
        spawn_chunk_x, spawn_chunk_z = self.chunk_manager.get_player_chunk(spawn_pos)
        
        # Request additional chunks around spawn for render distance
        chunks_in_range = self.chunk_manager.get_chunks_in_range(spawn_chunk_x, spawn_chunk_z)
        requested_count = 0
        
        for chunk_x, chunk_z in chunks_in_range:
            if (chunk_x, chunk_z) not in self.chunk_manager.loaded_chunks:
                if self.chunk_manager.request_chunk_load(chunk_x, chunk_z):
                    requested_count += 1
        
        print(f"Requested {requested_count} additional chunks around spawn ({spawn_chunk_x}, {spawn_chunk_z})")
        print("World Size: INFINITE (dynamic chunk loading with persistence)")
        print(f"Current render distance: {self.render_distance} chunks")
        print("Use +/- keys to adjust render distance (2-12 chunks)")
        print("Player modifications are now saved when chunks unload!")
    
    def load_texture(self):
        """Load all textures"""
        try:
            success = self.renderer.load_textures()
            if success:
                print("All textures loaded successfully")
                self.texture = self.renderer.block_texture
            else:
                print("Failed to load some textures")
                self.texture = None
        except Exception as e:
            print(f"Error loading textures: {e}")
            self.texture = None
    
    def handle_events(self):
        """Handle pygame events"""
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.running = False
            
            # Every printable key belongs to the picker's search box or to the
            # console while one of them is open: E types an 'e' there instead of
            # closing the window, and ESC is the way out of both.
            elif event.type == pg.TEXTINPUT and (self.console_open
                                                 or self.hud.inventory_open):
                # The key that opened the window arrives as KEYDOWN *and* as the
                # TEXTINPUT right behind it, by which time the window is already
                # open — swallow that one, or the search box starts with an 'e'
                # and the console with a second slash.
                if self._swallow_text:
                    self._swallow_text = False
                elif self.console_open:
                    self.console_text += event.text
                else:
                    self.hud.set_query(self.hud.query + event.text)
                    self.inv_hover = self.hud.hit_test(*pg.mouse.get_pos())

            elif event.type == pg.KEYDOWN:
                # F3 first on purpose: it is not a printable key, so it belongs
                # to the game even while the search box or the console has the
                # rest of the keyboard.
                if event.key == pg.K_F3:
                    self.show_debug = not self.show_debug
                elif self.console_open:
                    self.console_key(event.key)
                elif event.key == pg.K_ESCAPE:
                    if self.hud.inventory_open:
                        self.toggle_inventory()
                    else:
                        self.toggle_mouse_capture()
                elif self.hud.inventory_open:
                    if event.key == pg.K_BACKSPACE:
                        self.hud.set_query(self.hud.query[:-1])
                        self.inv_hover = self.hud.hit_test(*pg.mouse.get_pos())
                elif event.key == pg.K_e:
                    self.toggle_inventory()
                # T opens an empty line, / opens one with the slash already in
                # it — the real game's split, kept because every command here
                # starts with a slash and typing it twice is the usual slip.
                # Not with Ctrl or Alt held: those chords send no TEXTINPUT, so
                # the swallow set on opening would eat the first real character
                # typed instead of the one that opened the window.
                elif (event.key in CONSOLE_KEYS
                      and not event.mod & (pg.KMOD_CTRL | pg.KMOD_ALT | pg.KMOD_GUI)):
                    self.open_console('' if event.key == pg.K_t else '/')
                elif pg.K_1 <= event.key <= pg.K_1 + HOTBAR_SLOTS - 1:
                    self.select_slot(event.key - pg.K_1)
                # Render distance controls. /set renderdistance is the same
                # setter, so the two cannot disagree about the limits.
                elif event.key == pg.K_EQUALS or event.key == pg.K_KP_PLUS:
                    self.step_render_distance(1)
                elif event.key == pg.K_MINUS or event.key == pg.K_KP_MINUS:
                    self.step_render_distance(-1)
                elif event.key == pg.K_f:  # F key to toggle frustum culling
                    self.chunk_manager.toggle_frustum_culling()
                elif event.key == pg.K_TAB:  # TAB key to toggle flying mode
                    self.camera.toggle_flying()
                elif event.key == pg.K_SPACE:  # SPACE key to jump (in keydown for single press)
                    if not self.camera.flying:
                        self.camera.jump()
                elif event.key == pg.K_k:  # K key to toggle wireframe
                    self.renderer.toggle_wireframe()
            
            # The console holds the cursor as well as the keyboard: a click
            # would otherwise recapture the mouse and leave the line hanging
            # open with no way to see what you were typing at.
            elif event.type == pg.MOUSEBUTTONDOWN and not self.console_open:
                if self.hud.inventory_open:
                    if event.button == 1:
                        self.click_picker(event.pos)
                elif not self.mouse_captured:
                    self.capture_mouse()
                else:
                    # Handle block interaction when mouse is captured
                    if event.button == 1:  # Left click - remove block
                        self.remove_block()
                    elif event.button == 3:  # Right click - add block
                        self.add_block()

            elif event.type == pg.MOUSEBUTTONUP:
                self.dragging_scrollbar = False

            elif event.type == pg.MOUSEMOTION:
                if self.dragging_scrollbar:
                    self.hud.scroll_to_mouse(event.pos[1])
                    self.inv_hover = self.hud.hit_test(*event.pos)
                elif self.hud.inventory_open:
                    self.inv_hover = self.hud.hit_test(*event.pos)
                elif self.mouse_captured:
                    self.process_mouse_movement(event.rel[0], -event.rel[1])

            elif event.type == pg.MOUSEWHEEL:
                # The block list is longer than the panel, so while the picker
                # is open the wheel scrolls it; clicking a hotbar slot is how
                # you retarget in the meantime.
                if self.hud.inventory_open:
                    self.hud.scroll_by(-event.y * self.hud.scroll_step)
                    self.inv_hover = self.hud.hit_test(*pg.mouse.get_pos())
                elif not self.console_open:
                    # The console holds the cursor too, and its box is drawn
                    # right where the hotbar is: a nudge of the wheel while
                    # typing would retarget the slot invisibly, and the next
                    # right-click would place a block nobody chose.
                    self.select_slot((self.hotbar_slot - event.y) % HOTBAR_SLOTS)

            elif event.type == pg.VIDEORESIZE:
                self.renderer.resize(event.w, event.h)
                self.hud.resize(event.w, event.h)
    
    def capture_mouse(self):
        """Capture the mouse for camera control"""
        self.mouse_captured = True
        pg.mouse.set_visible(False)
        pg.event.set_grab(True)
        self.first_mouse = True
    
    def release_mouse(self):
        """Release the mouse"""
        self.mouse_captured = False
        pg.mouse.set_visible(True)
        pg.event.set_grab(False)
    
    def toggle_mouse_capture(self):
        """Toggle mouse capture state"""
        if self.mouse_captured:
            self.release_mouse()
        else:
            self.capture_mouse()

    def select_slot(self, slot):
        """Point the hotbar at *slot* and pick up whatever is in it."""
        self.hotbar_slot = slot
        self.selected_block_type = self.hud.hotbar[slot]

    def click_picker(self, pos):
        """Route a left click inside the open picker to whatever it landed on.

        Order matters only in that the grid is last: it is the one region that
        covers most of the panel, and the tab row, scrollbar and hotbar all sit
        outside it anyway.
        """
        tab = self.hud.tab_at(*pos)
        if tab >= 0:
            self.hud.set_tab(tab)
            self.inv_hover = self.hud.hit_test(*pos)
            return

        if self.hud.scrollbar_at(*pos):
            self.dragging_scrollbar = True
            self.hud.scroll_to_mouse(pos[1])
            return

        slot = self.hud.hotbar_slot_at(*pos)
        if slot >= 0:
            self.select_slot(slot)
            return

        # Picking a block fills the slot that is already selected, so the hotbar
        # stays visible underneath and you can see where it landed.
        block_id = self.hud.block_at(self.hud.hit_test(*pos))
        if block_id is not None:
            self.hud.set_slot(self.hotbar_slot, block_id)
            self.selected_block_type = block_id

    def toggle_inventory(self):
        """Open or close the creative block picker.

        The picker needs a visible cursor, so it borrows the mouse and hands it
        back in the state it found it — closing with ESC while the mouse had
        never been captured should not capture it.
        """
        self.hud.inventory_open = not self.hud.inventory_open

        if self.hud.inventory_open:
            self._was_captured = self.mouse_captured
            self._swallow_text = True
            self.release_mouse()
            pg.mouse.set_pos(self.renderer.width // 2, self.renderer.height // 2)
            self.inv_hover = self.hud.hit_test(*pg.mouse.get_pos())
        else:
            self.inv_hover = -1
            self.dragging_scrollbar = False
            self.hud.set_query('')
            if self._was_captured:
                self.capture_mouse()

    # ------------------------------------------------------------------
    # Command console — see commands.py for the commands themselves
    # ------------------------------------------------------------------

    CONSOLE_LINES = 8        # scrollback shown while the console is open
    CONSOLE_LINGER = 8.0     # seconds a printed line stays up after it closes

    def step_render_distance(self, delta):
        """The + and - keys. `/set renderdistance` is the same setter, and the
        limits live with it rather than in two places that can drift."""
        commands.set_render_distance(
            self, min(max(self.render_distance + delta, commands.RENDER_MIN),
                      commands.RENDER_MAX))

    def open_console(self, prefill=''):
        """Take the keyboard and the mouse, as the picker does, and hand both
        back in the state they were found in."""
        self.console_open = True
        self.console_text = self.console_draft = prefill
        self.console_at = len(self.console_history)
        self._swallow_text = True
        self._was_captured = self.mouse_captured
        self.release_mouse()

    def close_console(self):
        self.console_open = False
        self.console_text = ''
        if self._was_captured:
            self.capture_mouse()

    def console_key(self, key):
        """The non-printable half of the console's keyboard; the printable half
        arrives as TEXTINPUT, which is the only way to get a layout's own
        characters out of pygame."""
        if key in (pg.K_RETURN, pg.K_KP_ENTER):
            self.submit_console()
        elif key == pg.K_ESCAPE:
            self.close_console()
        elif key == pg.K_BACKSPACE:
            self.console_text = self.console_text[:-1]
        elif key == pg.K_UP:
            self.recall_console(-1)
        elif key == pg.K_DOWN:
            self.recall_console(1)

    def recall_console(self, delta):
        """UP and DOWN walk back through what has been submitted; one past the
        end is the line you were on before you started walking.

        Kept rather than blanked, because the line you started on is often not
        empty — `/` opens with a slash in it, and that is the one keystroke the
        `/` opener exists to save.
        """
        end = len(self.console_history)
        if delta < 0 and self.console_at == end:
            self.console_draft = self.console_text
        self.console_at = min(max(self.console_at + delta, 0), end)
        self.console_text = (self.console_draft if self.console_at == end
                             else self.console_history[self.console_at])

    def submit_console(self):
        """Run the line, keep what it printed, and close — the real game closes
        its chat on Enter too, which is why printed lines linger afterwards."""
        text = self.console_text.strip()
        self.close_console()
        if not text:
            return

        if not self.console_history or self.console_history[-1] != text:
            self.console_history.append(text)

        now = time.perf_counter()
        self.console_log += [(line, color, now)
                             for line, color in commands.dispatch(self, text)]
        del self.console_log[:-self.CONSOLE_LINES]   # nothing older is reachable

    def update_console(self):
        """Hand the HUD the lines that should be on screen.

        Runs every frame, because lines have to age out on their own — an
        unchanged console is one tuple comparison inside `set_console`.
        """
        cutoff = time.perf_counter() - self.CONSOLE_LINGER
        lines = [(text, color) for text, color, when in self.console_log
                 if self.console_open or when > cutoff]
        self.hud.set_console(
            lines[-self.CONSOLE_LINES:],
            '> ' + self.console_text + '|' if self.console_open else None)

    def process_mouse_movement(self, xoffset, yoffset):
        """Process mouse movement for camera control"""
        if self.first_mouse:
            self.first_mouse = False
            return
        
        xoffset *= self.mouse_sensitivity
        yoffset *= self.mouse_sensitivity
        
        self.camera.process_mouse_movement(xoffset, yoffset)
    
    def process_keyboard(self):
        """Process keyboard input for movement using strafe system like original main.py"""
        keys = pg.key.get_pressed()

        # Reset strafe state
        self.camera.strafe = [0, 0]

        # The picker and the console eat movement: WASD would otherwise walk the
        # player around behind a window they cannot see past, or spell a command
        # while walking. Gravity still runs.
        if self.hud.inventory_open or self.console_open:
            self.camera.sprinting = False
            return

        # Sprint status
        self.camera.sprinting = keys[pg.K_LSHIFT] or keys[pg.K_RSHIFT]
        
        # Set strafe based on key presses (fixed direction)
        if keys[pg.K_w]:
            self.camera.strafe[0] += 1.0  # Forward
        if keys[pg.K_s]:
            self.camera.strafe[0] -= 1.0  # Backward
        if keys[pg.K_a]:
            self.camera.strafe[1] -= 1.0  # Left
        if keys[pg.K_d]:
            self.camera.strafe[1] += 1.0  # Right
        
        # Flying mode vertical movement
        if self.camera.flying:
            if keys[pg.K_SPACE]:
                self.camera.position.y += self.movement_speed * self.delta_time
            if keys[pg.K_LSHIFT]:
                self.camera.position.y -= self.movement_speed * self.delta_time

    def update(self):
        """Update game state"""
        # Calculate delta time
        current_frame = time.time()
        self.delta_time = current_frame - self.last_frame
        self.last_frame = current_frame
        
        # Limit delta time to prevent physics issues
        self.delta_time = min(self.delta_time, 0.2)
        
        # Process input
        self.process_keyboard()
        
        # Update physics (gravity, movement, collision)
        # Process physics in smaller steps for stability (like original main.py)
        physics_steps = 8
        for _ in range(physics_steps):
            self.camera.update_physics(self.delta_time / physics_steps)
        
        # Update chunk loading based on player position
        self.chunk_manager.update(self.camera.position)

        self.update_debug()
        self.update_console()

    # The debug screen is assembled only while it is open, and then ten times a
    # second: get_chunk_info walks every loaded chunk for the LOD histogram,
    # which cost 0.9 ms a frame at render distance 24 back when the window title
    # carried these numbers. Ten times a second is as fast as the eye reads them.
    DEBUG_INTERVAL = 0.1        # seconds

    def update_debug(self):
        """Text for the F3 screen: where the player is, what they are looking
        at, and everything the window title used to report."""
        now = time.perf_counter()
        if not self.show_debug or now - self.last_debug_update < self.DEBUG_INTERVAL:
            return
        self.last_debug_update = now

        cam = self.camera
        pos = cam.position
        bx, by, bz = (int(math.floor(v)) for v in (pos.x, pos.y, pos.z))
        cx, cz = self.chunk_manager.world_to_chunk_coords(pos.x, pos.z)
        info = self.chunk_manager.get_chunk_info()
        look, axis = look_direction(cam.yaw)

        if cam.flying:
            mode = 'Flying'
        else:
            mode = 'Walking' if cam.on_ground else 'In air'
        if cam.sprinting:
            mode += ', sprinting'

        left = (
            'Minecraft Clone (ModernGL)',
            f'{self.clock.get_fps():.0f} fps ({self.delta_time * 1000:.1f} ms)',
            f'C: {self.rendered_chunks}/{self.total_chunks}   '
            f'F: {self.frustum_culled}   T: {self.triangles / 1000:.0f}k tri',
            f"L: {info['loaded_chunks']}   P: {info['pending_chunks']}   "
            f"Cache: {info['cached_chunks']}   E: {info['explored_chunks']}   "
            f"LOD: {'/'.join(str(n) for n in info['lod_counts'])}",
            f"RD: {self.render_distance}   frustum culling: "
            f"{'on' if info['frustum_culling'] else 'off'}   "
            f"wireframe: {'on' if self.renderer.wireframe_mode else 'off'}",
            '',
            f'XYZ: {pos.x:.3f} / {pos.y:.3f} / {pos.z:.3f}',
            f'Block: {bx} {by} {bz}',
            f'Chunk: {bx - cx * 16} {by} {bz - cz * 16} in {cx} {cz} '
            f'(LOD {self.chunk_manager.chunk_lod(cx, cz)})',
            f'Facing: {look} (towards {axis})   '
            f'({cam.yaw % 360:.1f} / {cam.pitch:.1f})',
            f'Biome: {BIOME_NAMES[column_biome(bx, bz)]}',
            f'{mode}   {math.hypot(cam.velocity.x, cam.velocity.z):.1f} m/s',
            f'Held: {BLOCK_NAMES.get(self.selected_block_type, "Unknown")}',
        )

        right = [f'Python {sys.version.split()[0]}   pygame {pg.version.ver}']
        if _PROC is not None:
            # cpu_percent() averages since its own previous call, and Windows
            # counts process CPU in 15.6 ms steps: sampled ten times a second it
            # quantises to zero. Once a second is a real number, and it is the
            # rate these are read at anyway.
            if now - self.last_cpu_sample >= 1.0:
                self.last_cpu_sample = now
                self.cpu_percent = _PROC.cpu_percent()
            right.append(f'Mem: {_PROC.memory_info().rss >> 20} MB   '
                         f'CPU: {self.cpu_percent:.0f}% of one core')
            right.append(f'RAM: {_RAM().percent:.0f}% of {_RAM_MB} MB')
        right += [f'Display: {self.renderer.width}x{self.renderer.height}',
                  self.gpu[0], f'GL {self.gpu[1]}', '']

        # The raycast render() already did — one frame old, which no eye can see
        # on a readout, and cheaper than casting the same ray twice.
        if self.target['hit']:
            block = int(self.target['block_type'])
            right += ['Targeted Block: {} {} {}'.format(*self.target['block_pos']),
                      f'{BLOCK_NAMES.get(block, "Unknown")} (id {block})']
        else:
            right.append('Targeted Block: none')

        self.hud.set_debug(left, tuple(right))


    def render(self):
        """Render the game"""
        # Clear screen
        self.renderer.clear()
        
        # Bind texture
        if self.texture:
            self.renderer.bind_texture(self.texture)
        
        # Update matrices
        view_matrix = self.camera.get_view_matrix()
        model_matrix = glm.mat4(1.0)
        self.renderer.update_matrices(view_matrix, model_matrix)
        
        # Render Sky (Background)
        elapsed_time = time.time() - self.start_time
        self.renderer.render_sky(view_matrix, elapsed_time)
        
        # Render chunks using chunk manager with frustum culling
        rendered_chunks, total_chunks, frustum_culled, triangles = self.chunk_manager.render_chunks(
            view_matrix, self.renderer.proj_matrix)

        # Store rendering stats for display
        self.rendered_chunks = rendered_chunks
        self.total_chunks = total_chunks
        self.frustum_culled = frustum_culled
        self.triangles = triangles
        
        # Render water surface after chunks (for proper transparency)
        self.renderer.render_water_surface(view_matrix, self.camera.position)
        
        # Block outline – draw around the block the player is looking at
        raycast_result = self.target = self.raycast()
        if raycast_result['hit']:
            self.renderer.render_block_outline(
                raycast_result['block_pos'], view_matrix, self.renderer.proj_matrix)
        
        # HUD: crosshair on top of everything
        self.renderer.render_crosshair()
        
        # Draw hotbar via OpenGL overlay
        self.hud.render(self.hotbar_slot, self.renderer.block_texture)

        # Creative picker sits on top of the hotbar, not instead of it
        if self.hud.inventory_open:
            self.hud.render_inventory(self.inv_hover, self.renderer.block_texture)

        # Above the picker, which dims the whole screen behind it
        self.hud.render_console()

        # Last, so it reads over the picker as well — the real F3 does the same
        if self.show_debug:
            self.hud.render_debug()

        # Swap buffers
        pg.display.flip()
    
    def run(self):
        """Main game loop"""
        print("Starting game loop...")
        
        while self.running:
            self.handle_events()
            self.update()
            self.render()
            self.clock.tick(60)  # 60 FPS
        
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up...")
        
        # Clean up chunks using chunk manager
        self.chunk_manager.cleanup()
        
        # Clean up texture
        if self.texture:
            self.texture.release()
        
        pg.quit()
        sys.exit()
    
    def raycast(self):
        """Cast a ray from camera to find the targeted block using 3D DDA algorithm"""
        origin = self.camera.position
        direction = self.camera.front
        max_distance = self.block_reach
        
        # Current voxel coordinates
        x = int(math.floor(origin.x))
        y = int(math.floor(origin.y))
        z = int(math.floor(origin.z))
        
        # Step direction for each axis (+1 or -1)
        step_x = 1 if direction.x > 0 else (-1 if direction.x < 0 else 0)
        step_y = 1 if direction.y > 0 else (-1 if direction.y < 0 else 0)
        step_z = 1 if direction.z > 0 else (-1 if direction.z < 0 else 0)
        
        # Distance along ray to cross first voxel boundary
        def get_t_max(p, d, step):
            if d == 0: return float('inf')
            boundary = math.floor(p) + (1.0 if step > 0 else 0.0)
            if step < 0 and p == boundary:
                boundary -= 1.0
            return (boundary - p) / d

        t_max_x = get_t_max(origin.x, direction.x, step_x)
        t_max_y = get_t_max(origin.y, direction.y, step_y)
        t_max_z = get_t_max(origin.z, direction.z, step_z)
        
        # How far to travel along ray to cross one voxel
        t_delta_x = abs(1.0 / direction.x) if direction.x != 0 else float('inf')
        t_delta_y = abs(1.0 / direction.y) if direction.y != 0 else float('inf')
        t_delta_z = abs(1.0 / direction.z) if direction.z != 0 else float('inf')
        
        # Keep track of previous coordinates so we know which face was hit
        prev_x, prev_y, prev_z = x, y, z
        
        while True:
            # Check if we hit a block
            block_type = self.get_block_at(x, y, z)
            if block_type not in REPLACEABLE:
                return {
                    'hit': True,
                    'block_pos': (x, y, z),
                    'prev_pos': (prev_x, prev_y, prev_z),
                    'block_type': block_type
                }
            
            # Save previous position to know which adjacent air block to place blocks in
            prev_x, prev_y, prev_z = x, y, z
            
            # Find the closest boundary and step into that voxel
            if t_max_x < t_max_y:
                if t_max_x < t_max_z:
                    if t_max_x > max_distance: break
                    x += step_x
                    t_max_x += t_delta_x
                else:
                    if t_max_z > max_distance: break
                    z += step_z
                    t_max_z += t_delta_z
            else:
                if t_max_y < t_max_z:
                    if t_max_y > max_distance: break
                    y += step_y
                    t_max_y += t_delta_y
                else:
                    if t_max_z > max_distance: break
                    z += step_z
                    t_max_z += t_delta_z
                    
        return {'hit': False}
    
    def get_block_at(self, x, y, z):
        """Get the block type at world coordinates"""
        return self.chunk_manager.get_block_at(x, y, z)
    
    def set_block_at(self, x, y, z, block_type):
        """Set the block type at world coordinates"""
        return self.chunk_manager.set_block_at(x, y, z, block_type)
    
    def remove_block(self):
        """Remove the block the player is looking at"""
        raycast_result = self.raycast()
        if raycast_result['hit']:
            x, y, z = raycast_result['block_pos']
            block = int(raycast_result['block_type'])
            self.set_block_at(x, y, z, AIR)

            # A door is two blocks and a tall plant is too. Taking one half and
            # leaving the other is the thing that looks broken, so both go.
            if block in UPPER:
                self.set_block_at(x, y + 1, z, AIR)
            elif block in LOWER:
                self.set_block_at(x, y - 1, z, AIR)

            print(f"Removed block at ({x}, {y}, {z})")

    def orient(self, block_id, cell, target):
        """The facing variant of *block_id* for this placement, or it unchanged.

        `ModernChunk.blocks` has one id per cell and no room for a state byte, so
        a door's four facings are four block ids (see blocks.FACING). Facing is
        which side of its own cell the geometry hugs: a ladder hugs the wall it
        was clicked onto, and everything else hugs the side the player is on, so
        a door faces them the way the real game's does.
        """
        row = FACING.get(block_id)
        if row is None:
            return block_id

        dx = target[0] - cell[0]
        dz = target[2] - cell[2]
        if block_id not in WALL_MOUNTED or (dx == 0 and dz == 0):
            yaw = math.radians(self.camera.yaw)
            dx, dz = -math.cos(yaw), -math.sin(yaw)

        if abs(dx) > abs(dz):
            return row[1] if dx > 0 else row[3]      # +X, -X
        return row[2] if dz > 0 else row[0]          # +Z, -Z

    def add_block(self):
        """Add a block next to the one the player is looking at"""
        raycast_result = self.raycast()
        if not raycast_result['hit']:
            return

        x, y, z = raycast_result['prev_pos']

        # Only place where there is nothing to displace — air, or water
        if self.get_block_at(x, y, z) not in REPLACEABLE:
            return

        block_id = self.orient(self.selected_block_type,
                               raycast_result['prev_pos'],
                               raycast_result['block_pos'])

        # Don't place a block the player is standing in — one they would then be
        # stuck inside. A torch or a flower is not one of those: the real game
        # lets you stand in them too, and refusing here is what made it
        # impossible to put a torch down at your own feet.
        if COLLIDES[block_id] and self.camera.intersects_block(x, y, z):
            return

        # A door is two blocks, and half a door is worse than none: if the upper
        # half has nowhere to go, nothing is placed at all.
        upper = UPPER.get(block_id)
        if upper is not None:
            if (self.get_block_at(x, y + 1, z) not in REPLACEABLE
                    or (COLLIDES[block_id]
                        and self.camera.intersects_block(x, y + 1, z))):
                return
            self.set_block_at(x, y + 1, z, upper)

        self.set_block_at(x, y, z, block_id)
        print(f"Placed {block_id} block at ({x}, {y}, {z})")


if __name__ == "__main__":
    try:
        # Check if required packages are available
        import moderngl
        import glm
        import numpy
        print("All required packages found!")
        
        # Create and run the game
        game = MinecraftModernGL(1200, 800)
        game.run()
        
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("Please install required packages:")
        print("pip install moderngl pygame numpy PyGLM numba")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting game: {e}")
        sys.exit(1)
