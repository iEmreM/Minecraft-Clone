import pygame as pg
import sys
import time
from engine.renderer import ModernGLRenderer
from engine.camera import Camera
from world.modern_chunk import ModernChunk, CHUNK_SIZE, CHUNK_HEIGHT, AIR, GRASS, DIRT, STONE, SAND, SNOW, LEAVES, WOOD, WATER
from world.threaded_chunk_manager import ThreadedChunkManager
import glm
import math


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
        
        # Movement
        self.movement_speed = 20.0
        self.mouse_sensitivity = 0.3
        
        # Block interaction
        self.block_reach = 8.0  # How far the player can reach
        self.selected_block_type = 1  # GRASS by default
        
        # World state with threaded chunk manager
        self.render_distance = 6  # Configurable render distance
        self.chunk_manager = ThreadedChunkManager(self.renderer, self.render_distance)
        self.texture = None
        
        # Set world reference for camera collision detection
        self.camera.set_world(self.chunk_manager)
        
        self.initialize_spawn_chunks()
        
        # Load texture
        self.load_texture()
        
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
        print("- O: Toggle occlusion culling")
        print("- C: Toggle conservative occlusion mode")
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
                # Block selection
                elif event.key == pg.K_1:
                    self.selected_block_type = GRASS
                elif event.key == pg.K_2:
                    self.selected_block_type = DIRT
                elif event.key == pg.K_3:
                    self.selected_block_type = STONE
                elif event.key == pg.K_4:
                    self.selected_block_type = SAND
                elif event.key == pg.K_5:
                    self.selected_block_type = SNOW
                elif event.key == pg.K_6:
                    self.selected_block_type = LEAVES
                elif event.key == pg.K_7:
                    self.selected_block_type = WOOD
                elif event.key == pg.K_8:
                    self.selected_block_type = WATER
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
                elif event.key == pg.K_o:  # O key to toggle occlusion culling
                    self.chunk_manager.toggle_occlusion_culling()
                elif event.key == pg.K_c:  # C key to toggle conservative occlusion mode
                    self.chunk_manager.toggle_occlusion_conservative_mode()
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
            
            elif event.type == pg.VIDEORESIZE:
                self.renderer.resize(event.w, event.h)
    
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
        
        # Set strafe based on key presses (fixed direction)
        if keys[pg.K_w]:
            self.camera.strafe[0] += 1.25  # Forward
        if keys[pg.K_s]:
            self.camera.strafe[0] -= 1.25  # Backward
        if keys[pg.K_a]:
            self.camera.strafe[1] -= 1.25  # Left
        if keys[pg.K_d]:
            self.camera.strafe[1] += 1.25  # Right
        
        # Flying mode vertical movement
        if self.camera.flying:
            if keys[pg.K_SPACE]:
                self.camera.position.y += self.movement_speed * self.delta_time
            if keys[pg.K_LSHIFT]:
                self.camera.position.y -= self.movement_speed * self.delta_time
        
        # Block type selection
        if keys[pg.K_1]:
            self.selected_block_type = GRASS
        if keys[pg.K_2]:
            self.selected_block_type = DIRT
        if keys[pg.K_3]:
            self.selected_block_type = STONE
        if keys[pg.K_4]:
            self.selected_block_type = SAND
        if keys[pg.K_5]:
            self.selected_block_type = SNOW
        if keys[pg.K_6]:
            self.selected_block_type = LEAVES
        if keys[pg.K_7]:
            self.selected_block_type = WOOD
        
        # Render distance controls - check for key press events, not held keys
        # These need to be handled in handle_events instead
    
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
        
        # Update window title with FPS and camera position
        fps = self.clock.get_fps()
        pos = self.camera.position
        block_names = {GRASS: "Grass", DIRT: "Dirt", STONE: "Stone", SAND: "Sand", SNOW: "Snow", LEAVES: "Leaves", WOOD: "Wood", WATER: "Water"}
        selected_name = block_names.get(self.selected_block_type, "Unknown")
        chunk_info = self.chunk_manager.get_chunk_info()
        chunks_loaded = chunk_info['loaded_chunks']
        pending_chunks = chunk_info['pending_chunks']
        cached_chunks = chunk_info['cached_chunks']
        explored_chunks = chunk_info['explored_chunks']
        frustum_enabled = chunk_info['frustum_culling']
        occlusion_enabled = chunk_info['occlusion_culling']
        
        # Include culling stats if available
        if hasattr(self, 'rendered_chunks') and hasattr(self, 'total_chunks'):
            culling_info = f"L:{chunks_loaded} C:{cached_chunks} E:{explored_chunks} R:{self.rendered_chunks}/{self.total_chunks}"
            if hasattr(self, 'frustum_culled') and hasattr(self, 'occlusion_culled'):
                culling_info += f" (F:{self.frustum_culled} O:{self.occlusion_culled})"
        else:
            culling_info = f"L:{chunks_loaded} C:{cached_chunks} E:{explored_chunks} P:{pending_chunks}"
        
        frustum_status = "FC:ON" if frustum_enabled else "FC:OFF"
        occlusion_status = "OC:ON" if occlusion_enabled else "OC:OFF"
        flying_status = "FLY" if self.camera.flying else "WALK"
        pg.display.set_caption(f'Minecraft ModernGL - FPS: {fps:.0f} | Pos: ({pos.x:.1f}, {pos.y:.1f}, {pos.z:.1f}) | {flying_status} | Block: {selected_name} | Chunks: {culling_info} | RD: {self.render_distance} | {frustum_status} | {occlusion_status}')
    
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
        
        # Render chunks using chunk manager with frustum and occlusion culling
        rendered_chunks, total_chunks, frustum_culled, occlusion_culled = self.chunk_manager.render_chunks(
            view_matrix, self.renderer.proj_matrix, self.camera.position)
        
        # Store rendering stats for display
        self.rendered_chunks = rendered_chunks
        self.total_chunks = total_chunks
        self.frustum_culled = frustum_culled
        self.occlusion_culled = occlusion_culled
        
        # Render water surface after chunks (for proper transparency)
        self.renderer.render_water_surface(view_matrix, self.camera.position)
        
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
        """Cast a ray from camera to find the targeted block"""
        # Ray origin (camera position)
        origin = self.camera.position
        
        # Ray direction (camera front vector)
        direction = self.camera.front
        
        # Step along the ray to find block intersection
        step_size = 0.1
        max_distance = self.block_reach
        
        current_pos = glm.vec3(origin)
        
        for i in range(int(max_distance / step_size)):
            # Move along the ray
            current_pos += direction * step_size
            
            # Convert to block coordinates
            block_x = int(math.floor(current_pos.x))
            block_y = int(math.floor(current_pos.y))
            block_z = int(math.floor(current_pos.z))
            
            # Check if we hit a block
            block_type = self.get_block_at(block_x, block_y, block_z)
            if block_type != AIR:
                # Calculate the previous position (for block placement)
                prev_pos = current_pos - direction * step_size
                prev_block_x = int(math.floor(prev_pos.x))
                prev_block_y = int(math.floor(prev_pos.y))
                prev_block_z = int(math.floor(prev_pos.z))
                
                return {
                    'hit': True,
                    'block_pos': (block_x, block_y, block_z),
                    'prev_pos': (prev_block_x, prev_block_y, prev_block_z),
                    'block_type': block_type
                }
        
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
            
            # Don't place block if it would intersect with the player
            player_pos = self.camera.position
            if (abs(player_pos.x - x) < 1.5 and 
                abs(player_pos.y - y) < 2.0 and 
                abs(player_pos.z - z) < 1.5):
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
