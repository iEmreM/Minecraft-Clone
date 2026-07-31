import pygame as pg
import sys
import time
from engine.renderer import ModernGLRenderer
from engine.camera import Camera
from world.modern_chunk import ModernChunk, CHUNK_SIZE, CHUNK_HEIGHT, AIR, GRASS, DIRT, STONE, SAND, SNOW, LEAVES, WOOD, WATER, STONE_BRICK, BRICK
from world.threaded_chunk_manager import ThreadedChunkManager
import glm
import math
from engine.hud import HUDRenderer, HOTBAR_BLOCKS


class MinecraftModernGL:
    def __init__(self, width=800, height=600):
        # Initialize renderer
        self.renderer = ModernGLRenderer(width, height)
        
        # Initialize camera - position it above the terrain 
        self.camera = Camera(position=(8, 40, 8))
        
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
        self.last_title_update = 0.0
        
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
        self.hud = HUDRenderer(self.renderer.ctx, 1200, 800)
        
        print("Minecraft ModernGL initialized successfully!")
        print("Controls:")
        print("- WASD: Move (walk/fly)")
        print("- Mouse: Look around")
        print("- Space: Jump (walking) / Up (flying)")
        print("- Shift: Down (flying only)")
        print("- TAB: Toggle flying mode")
        print("- ESC: Toggle mouse capture")
        print("- Click to capture mouse")
        print("- Left Click: Remove block")
        print("- Right Click: Place block")
        print("- 1-8: Select block type (Grass/Dirt/Stone/Sand/Snow/Leaves/Wood/Water)")
        print("- +/-: Increase/Decrease render distance")
        print("- F: Toggle frustum culling")
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
            
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    self.toggle_mouse_capture()
                elif event.key == pg.K_1:
                    self.hotbar_slot = 0
                    self.selected_block_type = HOTBAR_BLOCKS[0]
                elif event.key == pg.K_2:
                    self.hotbar_slot = 1
                    self.selected_block_type = HOTBAR_BLOCKS[1]
                elif event.key == pg.K_3:
                    self.hotbar_slot = 2
                    self.selected_block_type = HOTBAR_BLOCKS[2]
                elif event.key == pg.K_4:
                    self.hotbar_slot = 3
                    self.selected_block_type = HOTBAR_BLOCKS[3]
                elif event.key == pg.K_5:
                    self.hotbar_slot = 4
                    self.selected_block_type = HOTBAR_BLOCKS[4]
                elif event.key == pg.K_6:
                    self.hotbar_slot = 5
                    self.selected_block_type = HOTBAR_BLOCKS[5]
                elif event.key == pg.K_7:
                    self.hotbar_slot = 6
                    self.selected_block_type = HOTBAR_BLOCKS[6]
                elif event.key == pg.K_8:
                    self.hotbar_slot = 7
                    self.selected_block_type = HOTBAR_BLOCKS[7]
                elif event.key == pg.K_9:
                    self.hotbar_slot = 8
                    self.selected_block_type = HOTBAR_BLOCKS[8]
                # Render distance controls
                elif event.key == pg.K_EQUALS or event.key == pg.K_KP_PLUS:  # + key
                    new_distance = min(self.render_distance + 1, 96)  # Max 96 chunks
                    if new_distance != self.render_distance:
                        self.render_distance = new_distance
                        self.chunk_manager.set_render_distance(new_distance)
                        print(f"Render distance increased to: {new_distance}")
                elif event.key == pg.K_MINUS or event.key == pg.K_KP_MINUS:  # - key
                    new_distance = max(self.render_distance - 1, 2)  # Min 2 chunks
                    if new_distance != self.render_distance:
                        self.render_distance = new_distance
                        self.chunk_manager.set_render_distance(new_distance)
                        print(f"Render distance decreased to: {new_distance}")
                elif event.key == pg.K_f:  # F key to toggle frustum culling
                    self.chunk_manager.toggle_frustum_culling()
                elif event.key == pg.K_TAB:  # TAB key to toggle flying mode
                    self.camera.toggle_flying()
                elif event.key == pg.K_SPACE:  # SPACE key to jump (in keydown for single press)
                    if not self.camera.flying:
                        self.camera.jump()
                elif event.key == pg.K_k:  # K key to toggle wireframe
                    self.renderer.toggle_wireframe()
            
            elif event.type == pg.MOUSEBUTTONDOWN:
                if not self.mouse_captured:
                    self.capture_mouse()
                else:
                    # Handle block interaction when mouse is captured
                    if event.button == 1:  # Left click - remove block
                        self.remove_block()
                    elif event.button == 3:  # Right click - add block
                        self.add_block()
            
            elif event.type == pg.MOUSEMOTION:
                if self.mouse_captured:
                    self.process_mouse_movement(event.rel[0], -event.rel[1])
            
            elif event.type == pg.MOUSEWHEEL:
                self.hotbar_slot = (self.hotbar_slot - event.y) % len(HOTBAR_BLOCKS)
                self.selected_block_type = HOTBAR_BLOCKS[self.hotbar_slot]
            
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

        self.update_title()

    # Telemetry is a window title, so it only has to keep up with the eye.
    # Rebuilding it every frame cost 0.9 ms at render distance 24, nearly all of
    # it get_chunk_info walking every loaded chunk for the LOD histogram.
    TITLE_INTERVAL = 0.1        # seconds

    def update_title(self):
        """Window title: FPS, position, chunk and LOD counts, live triangles."""
        now = time.perf_counter()
        if now - self.last_title_update < self.TITLE_INTERVAL:
            return
        self.last_title_update = now

        fps = self.clock.get_fps()
        pos = self.camera.position
        block_names = {
            GRASS: "Grass", DIRT: "Dirt", STONE: "Stone", SAND: "Sand",
            SNOW: "Snow", LEAVES: "Leaves", WOOD: "Wood", WATER: "Water",
            STONE_BRICK: "Stone Brick", BRICK: "Brick"
        }
        selected_name = block_names.get(self.selected_block_type, "Unknown")
        chunk_info = self.chunk_manager.get_chunk_info()
        chunks_loaded = chunk_info['loaded_chunks']
        pending_chunks = chunk_info['pending_chunks']
        cached_chunks = chunk_info['cached_chunks']
        explored_chunks = chunk_info['explored_chunks']
        frustum_enabled = chunk_info['frustum_culling']

        # Include culling stats if available
        if hasattr(self, 'rendered_chunks') and hasattr(self, 'total_chunks'):
            culling_info = f"L:{chunks_loaded} C:{cached_chunks} E:{explored_chunks} R:{self.rendered_chunks}/{self.total_chunks}"
            if hasattr(self, 'frustum_culled'):
                culling_info += f" (F:{self.frustum_culled})"
        else:
            culling_info = f"L:{chunks_loaded} C:{cached_chunks} E:{explored_chunks} P:{pending_chunks}"

        frustum_status = "FC:ON" if frustum_enabled else "FC:OFF"

        # Chunks per detail level, then live triangles — the two numbers that
        # say whether the LOD rings are actually doing anything.
        lod_info = "/".join(str(count) for count in chunk_info['lod_counts'])
        triangle_info = f"{getattr(self, 'triangles', 0) / 1000:.0f}k"

        if self.camera.flying:
            move_status = "FLY(SPRINT)" if self.camera.sprinting else "FLY"
        else:
            move_status = "SPRINT" if self.camera.sprinting else "WALK"
            
        pg.display.set_caption(f'Minecraft ModernGL - FPS: {fps:.0f} | Pos: ({pos.x:.1f}, {pos.y:.1f}, {pos.z:.1f}) | {move_status} | Block: {selected_name} | Chunks: {culling_info} | LOD: {lod_info} | Tri: {triangle_info} | RD: {self.render_distance} | {frustum_status}')
    
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
        raycast_result = self.raycast()
        if raycast_result['hit']:
            self.renderer.render_block_outline(
                raycast_result['block_pos'], view_matrix, self.renderer.proj_matrix)
        
        # HUD: crosshair on top of everything
        self.renderer.render_crosshair()
        
        # Draw hotbar via OpenGL overlay
        self.hud.render(self.hotbar_slot, self.renderer.block_texture)
        
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
            if block_type != AIR:
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
            self.set_block_at(x, y, z, AIR)
            print(f"Removed block at ({x}, {y}, {z})")
    
    def add_block(self):
        """Add a block next to the one the player is looking at"""
        raycast_result = self.raycast()
        if raycast_result['hit']:
            x, y, z = raycast_result['prev_pos']
            
            # Don't place a block the player is standing in
            if self.camera.intersects_block(x, y, z):
                return

            # Only place if the position is currently air
            if self.get_block_at(x, y, z) == AIR:
                self.set_block_at(x, y, z, self.selected_block_type)
                print(f"Placed {self.selected_block_type} block at ({x}, {y}, {z})")


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
