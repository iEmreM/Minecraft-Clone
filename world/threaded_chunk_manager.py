import math
import glm
import threading
import queue
import time
from world.modern_chunk import ModernChunk
from engine.frustum import Frustum
from engine.occlusion import OcclusionCuller

class ThreadedChunkManager:
    """Manages dynamic chunk loading and unloading with background threading to eliminate lag"""
    
    def __init__(self, renderer, render_distance=8):
        self.renderer = renderer
        self.render_distance = render_distance
        self.chunks = {}  # Dictionary of (x, z) -> chunk (main thread access)
        self.loaded_chunks = set()  # Set of (x, z) coordinates for loaded chunks
        self.last_player_chunk = None
        
        # Chunk persistence system
        self.chunk_cache = {}  # Cache for unloaded but persistent chunks
        self.explored_chunks = set()  # Set of chunk coordinates that have been generated
        
        # Pre-generation settings
        self.initial_chunks_generated = False
        self.chunks_to_pregenerate = 20  # Number of chunks around spawn to generate
        
        # Threading components
        self.loading_thread = None
        self.should_stop = False
        self.chunk_queue = queue.Queue()  # Queue for chunk operations
        self.completed_chunks = queue.Queue()  # Completed chunks ready for main thread
        self.chunks_to_unload = queue.Queue()  # Chunks to be unloaded
        self.thread_lock = threading.Lock()
        
        # Async mesh building queues
        self.mesh_build_queue = queue.PriorityQueue()  # Priority queue - closer chunks processed first
        self.completed_meshes = queue.Queue()  # Completed mesh data ready for VAO creation
        self.player_position = glm.vec3(0, 0, 0)  # Track player position for priority calculation
        self.mesh_request_counter = 0  # Counter for tiebreaker in priority queue
        
        # Frustum culling
        self.frustum = Frustum()
        self.enable_frustum_culling = True
        
        # Occlusion culling
        self.occlusion_culler = OcclusionCuller()
        self.enable_occlusion_culling = True
        
        # Conservative mode by default to prevent over-culling
        self.occlusion_culler.set_conservative_mode(True)
        
        # Start background thread
        self.start_background_thread()
        
        print(f"ThreadedChunkManager initialized with render distance: {render_distance}")

    
    def pregenerate_spawn_chunks(self, spawn_x, spawn_z):
        """Pre-generate chunks around spawn position before game starts"""
        if self.initial_chunks_generated:
            return
        
        print(f"Pre-generating {self.chunks_to_pregenerate} chunks around spawn ({spawn_x}, {spawn_z})...")
        
        # Calculate spawn chunk coordinates
        from world.modern_chunk import CHUNK_SIZE
        spawn_chunk_x = int(spawn_x // CHUNK_SIZE)
        spawn_chunk_z = int(spawn_z // CHUNK_SIZE)
        
        # Generate chunks in a square around spawn
        radius = int(math.sqrt(self.chunks_to_pregenerate) // 2) + 1
        generated_count = 0
        
        for x in range(spawn_chunk_x - radius, spawn_chunk_x + radius + 1):
            for z in range(spawn_chunk_z - radius, spawn_chunk_z + radius + 1):
                if generated_count >= self.chunks_to_pregenerate:
                    break
                
                # Generate chunk immediately (synchronously for pre-gen)
                chunk = ModernChunk(x, z, self.renderer, chunk_data=None, chunk_manager=self)
                
                # Build mesh synchronously for pre-generated chunks so they're ready immediately
                chunk.build_mesh()
                
                self.chunks[(x, z)] = chunk
                self.loaded_chunks.add((x, z))
                self.explored_chunks.add((x, z))
                generated_count += 1
                
                print(f"Pre-generated chunk ({x}, {z}) - {generated_count}/{self.chunks_to_pregenerate}")
            
            if generated_count >= self.chunks_to_pregenerate:
                break
        
        self.initial_chunks_generated = True
        print(f"Pre-generation complete! Generated {generated_count} chunks with meshes ready.")
    
    def save_chunk_to_cache(self, chunk_x, chunk_z):
        """Save chunk data to cache before unloading"""
        if (chunk_x, chunk_z) in self.chunks:
            chunk = self.chunks[(chunk_x, chunk_z)]
            self.chunk_cache[(chunk_x, chunk_z)] = chunk.save_chunk_data()
            self.explored_chunks.add((chunk_x, chunk_z))
            return True
        return False
    
    def load_chunk_from_cache(self, chunk_x, chunk_z):
        """Load chunk data from cache if available"""
        if (chunk_x, chunk_z) in self.chunk_cache:
            chunk_data = self.chunk_cache[(chunk_x, chunk_z)]
            chunk = ModernChunk(chunk_x, chunk_z, self.renderer, chunk_data, chunk_manager=self)
            return chunk
        return None
    
    def is_chunk_explored(self, chunk_x, chunk_z):
        """Check if a chunk has been previously generated/explored"""
        return (chunk_x, chunk_z) in self.explored_chunks
    
    def start_background_thread(self):
        """Start the background chunk loading thread"""
        self.loading_thread = threading.Thread(target=self._chunk_worker, daemon=True)
        self.loading_thread.start()
    
    def _chunk_worker(self):
        """Background thread worker for chunk loading and mesh building"""
        while not self.should_stop:
            try:
                # Check for chunk loading requests
                try:
                    operation = self.chunk_queue.get(timeout=0.1)
                    if operation['type'] == 'load':
                        chunk_x, chunk_z = operation['coords']
                        
                        # Try to load from cache first
                        chunk = self.load_chunk_from_cache(chunk_x, chunk_z)
                        if chunk is None:
                            # Create new chunk if not in cache
                            chunk = ModernChunk(chunk_x, chunk_z, self.renderer, chunk_data=None, chunk_manager=self)
                            self.explored_chunks.add((chunk_x, chunk_z))
                        
                        # Queue it for main thread integration
                        self.completed_chunks.put({
                            'type': 'loaded',
                            'coords': (chunk_x, chunk_z),
                            'chunk': chunk
                        })
                    elif operation['type'] == 'unload':
                        chunk_x, chunk_z = operation['coords']
                        self.chunks_to_unload.put((chunk_x, chunk_z))
                except queue.Empty:
                    pass
                
                # Check for mesh building requests (priority queue format)
                try:
                    priority_item = self.mesh_build_queue.get(timeout=0.01)
                    # PriorityQueue returns (priority, counter, data) tuple
                    priority, counter, mesh_request = priority_item
                    chunk_coords = mesh_request['coords']
                    chunk = mesh_request['chunk']
                    
                    # Build mesh in background thread
                    from world.fast_builder import build_chunk_mesh_fast
                    vertices_array, indices_array = build_chunk_mesh_fast(
                        chunk.blocks, chunk.chunk_x, chunk.chunk_z
                    )
                    
                    # Queue completed mesh for main thread
                    self.completed_meshes.put({
                        'coords': chunk_coords,
                        'vertices': vertices_array,
                        'indices': indices_array
                    })
                except queue.Empty:
                    pass
                    
            except Exception as e:
                print(f"Error in chunk worker thread: {e}")
                time.sleep(0.1)
    
    def _calculate_chunk_priority(self, chunk_x, chunk_z):
        """Calculate priority for chunk based on distance to player (lower = higher priority)"""
        from world.modern_chunk import CHUNK_SIZE
        # Calculate chunk center world position
        chunk_center_x = (chunk_x * CHUNK_SIZE) + (CHUNK_SIZE / 2)
        chunk_center_z = (chunk_z * CHUNK_SIZE) + (CHUNK_SIZE / 2)
        
        # Calculate distance to player (use 2D distance, ignore Y)
        dx = chunk_center_x - self.player_position.x
        dz = chunk_center_z - self.player_position.z
        distance = math.sqrt(dx*dx + dz*dz)
        
        return distance  # Lower distance = higher priority
    
    def clear_distant_mesh_requests(self):
        """Clear mesh requests for chunks that are now too far from player"""
        from world.modern_chunk import CHUNK_SIZE
        current_chunk_x = int(self.player_position.x // CHUNK_SIZE)
        current_chunk_z = int(self.player_position.z // CHUNK_SIZE)
        
        # Create new queue with only relevant chunks
        new_queue = queue.PriorityQueue()
        cleared_count = 0
        kept_count = 0
        
        # Drain existing queue
        while True:
            try:
                priority_item = self.mesh_build_queue.get_nowait()
                priority, counter, mesh_request = priority_item
                chunk_x, chunk_z = mesh_request['coords']
                
                # Calculate chunk distance in chunk coordinates
                chunk_dx = abs(chunk_x - current_chunk_x)
                chunk_dz = abs(chunk_z - current_chunk_z)
                
                # Only keep chunks within render distance
                if chunk_dx <= self.render_distance and chunk_dz <= self.render_distance:
                    # Recalculate priority with current player position
                    new_priority = self._calculate_chunk_priority(chunk_x, chunk_z)
                    new_queue.put((new_priority, counter, mesh_request))
                    kept_count += 1
                else:
                    cleared_count += 1
            except queue.Empty:
                break
        
        # Replace queue
        self.mesh_build_queue = new_queue
        
        if cleared_count > 0:
            print(f"Cleared {cleared_count} distant mesh requests, kept {kept_count}")
    
    def world_to_chunk_coords(self, world_x, world_z):
        """Convert world coordinates to chunk coordinates"""
        from world.modern_chunk import CHUNK_SIZE
        chunk_x = int(world_x // CHUNK_SIZE)
        chunk_z = int(world_z // CHUNK_SIZE)
        return chunk_x, chunk_z
    
    def get_player_chunk(self, player_pos):
        """Get the chunk coordinates the player is currently in"""
        return self.world_to_chunk_coords(player_pos.x, player_pos.z)
    
    def get_chunks_in_range(self, center_chunk_x, center_chunk_z):
        """Get all chunk coordinates within render distance of center chunk"""
        chunks_in_range = set()
        
        for x in range(center_chunk_x - self.render_distance, 
                      center_chunk_x + self.render_distance + 1):
            for z in range(center_chunk_z - self.render_distance, 
                          center_chunk_z + self.render_distance + 1):
                # Use circular render distance
                distance = math.sqrt((x - center_chunk_x)**2 + (z - center_chunk_z)**2)
                if distance <= self.render_distance:
                    chunks_in_range.add((x, z))
        
        return chunks_in_range
    
    def request_chunk_load(self, chunk_x, chunk_z):
        """Request a chunk to be loaded in the background"""
        if (chunk_x, chunk_z) not in self.chunks and (chunk_x, chunk_z) not in self.loaded_chunks:
            self.chunk_queue.put({
                'type': 'load',
                'coords': (chunk_x, chunk_z)
            })
            # Mark as pending to avoid duplicate requests
            self.loaded_chunks.add((chunk_x, chunk_z))
            return True
        return False
    
    def request_chunk_unload(self, chunk_x, chunk_z):
        """Request a chunk to be unloaded"""
        if (chunk_x, chunk_z) in self.chunks:
            self.chunk_queue.put({
                'type': 'unload',
                'coords': (chunk_x, chunk_z)
            })
            return True
        return False
    
    def process_completed_chunks(self):
        """Process chunks that have been loaded in the background (call from main thread)"""
        processed = 0
        max_per_frame = 12  # Increased for faster updates when player moves quickly
        
        while processed < max_per_frame:
            try:
                result = self.completed_chunks.get_nowait()
                if result['type'] == 'loaded':
                    chunk_x, chunk_z = result['coords']
                    chunk = result['chunk']
                    
                    # Check if chunk has cached mesh - if so, create VAO immediately
                    if chunk.mesh_cache_valid and chunk.cached_vertices is not None:
                        # Create VAO from cached mesh on main thread
                        vertices = chunk.cached_vertices
                        indices = chunk.cached_indices
                        
                        if len(vertices) > 0:
                            if chunk.vao:
                                chunk.vao.release()
                            chunk.vao = self.renderer.create_vao(vertices, indices)
                            chunk.vertex_count = len(indices)
                        
                        chunk.needs_update = False
                    elif chunk.needs_update:
                        # No cached mesh - request mesh building in background with priority
                        priority = self._calculate_chunk_priority(chunk_x, chunk_z)
                        self.mesh_request_counter += 1
                        self.mesh_build_queue.put((priority, self.mesh_request_counter, {
                            'coords': (chunk_x, chunk_z),
                            'chunk': chunk
                        }))
                    
                    # Add chunk to loaded chunks now that it's ready (or mesh is building)
                    with self.thread_lock:
                        self.chunks[(chunk_x, chunk_z)] = chunk
                    
                    processed += 1
                    
            except queue.Empty:
                break
        
        # Process completed meshes and create VAOs (main thread only for OpenGL)
        mesh_processed = 0
        while mesh_processed < max_per_frame:
            try:
                mesh_data = self.completed_meshes.get_nowait()
                chunk_coords = mesh_data['coords']
                vertices = mesh_data['vertices']
                indices = mesh_data['indices']
                
                # Get the chunk and create VAO on main thread
                with self.thread_lock:
                    if chunk_coords in self.chunks:
                        chunk = self.chunks[chunk_coords]
                        
                        # Update chunk's cached mesh data
                        chunk.cached_vertices = vertices
                        chunk.cached_indices = indices
                        chunk.mesh_cache_valid = True
                        
                        # Create VAO on main thread (OpenGL requirement)
                        if len(vertices) > 0:
                            if chunk.vao:
                                chunk.vao.release()
                            chunk.vao = self.renderer.create_vao(vertices, indices)
                            chunk.vertex_count = len(indices)
                        
                        chunk.needs_update = False
                        mesh_processed += 1
            except queue.Empty:
                break
        
        # Process unload requests
        unload_count = 0
        while unload_count < max_per_frame:
            try:
                chunk_x, chunk_z = self.chunks_to_unload.get_nowait()
                self.unload_chunk_immediate(chunk_x, chunk_z)
                unload_count += 1
            except queue.Empty:
                break
        
        return processed, unload_count
    
    def unload_chunk_immediate(self, chunk_x, chunk_z):
        """Immediately unload a chunk (called from main thread)"""
        if (chunk_x, chunk_z) in self.chunks:
            chunk = self.chunks[(chunk_x, chunk_z)]
            
            # Save chunk data to cache before unloading
            self.save_chunk_to_cache(chunk_x, chunk_z)
            
            # Clean up GPU resources
            if hasattr(chunk, 'vao') and chunk.vao:
                chunk.vao.release()
            
            # Remove from dictionaries
            with self.thread_lock:
                del self.chunks[(chunk_x, chunk_z)]
            self.loaded_chunks.discard((chunk_x, chunk_z))
            return True
        return False
    
    def update(self, player_pos):
        """Update chunk loading/unloading based on player position"""
        # Update player position for mesh priority calculation
        old_position = self.player_position
        self.player_position = player_pos
        
        current_chunk = self.get_player_chunk(player_pos)
        
        # Process any completed chunks first (increased limit for faster updates)
        loaded, unloaded = self.process_completed_chunks()
        
        # Only update if player moved to a different chunk
        if current_chunk != self.last_player_chunk:
            # Calculate movement distance
            if self.last_player_chunk is not None:
                chunk_moved = max(
                    abs(current_chunk[0] - self.last_player_chunk[0]),
                    abs(current_chunk[1] - self.last_player_chunk[1])
                )
                
                # If player moved more than 2 chunks, clear distant mesh requests
                if chunk_moved >= 2:
                    self.clear_distant_mesh_requests()
            
            self.last_player_chunk = current_chunk
            
            # Get chunks that should be loaded
            chunks_to_load = self.get_chunks_in_range(current_chunk[0], current_chunk[1])
            
            # Request loading of new chunks
            load_requests = 0
            for chunk_x, chunk_z in chunks_to_load:
                if self.request_chunk_load(chunk_x, chunk_z):
                    load_requests += 1
            
            # Request unloading of chunks that are too far away
            unload_requests = 0
            for chunk_coords in list(self.loaded_chunks):
                if chunk_coords not in chunks_to_load:
                    if self.request_chunk_unload(chunk_coords[0], chunk_coords[1]):
                        unload_requests += 1
            
            if load_requests > 0 or unload_requests > 0:
                print(f"Player moved to chunk {current_chunk}. "
                      f"Load requests: {load_requests}, Unload requests: {unload_requests}, "
                      f"Total chunks: {len(self.chunks)}")
        
        return loaded > 0 or unloaded > 0  # Return True if any changes occurred
    
    def get_chunk(self, chunk_x, chunk_z):
        """Get a chunk at the given coordinates, or None if not loaded"""
        with self.thread_lock:
            return self.chunks.get((chunk_x, chunk_z))
    
    def get_block_at(self, world_x, world_y, world_z):
        """Get block type at world coordinates"""
        chunk_x, chunk_z = self.world_to_chunk_coords(world_x, world_z)
        chunk = self.get_chunk(chunk_x, chunk_z)
        
        if chunk:
            from world.modern_chunk import CHUNK_SIZE
            # Convert world coordinates to local chunk coordinates
            local_x = int(world_x - chunk_x * CHUNK_SIZE)
            local_z = int(world_z - chunk_z * CHUNK_SIZE)
            return chunk.get_block(local_x, int(world_y), local_z)
        
        return 0  # AIR if chunk not loaded
    
    def set_block_at(self, world_x, world_y, world_z, block_type):
        """Set block type at world coordinates"""
        chunk_x, chunk_z = self.world_to_chunk_coords(world_x, world_z)
        chunk = self.get_chunk(chunk_x, chunk_z)
        
        if chunk:
            from world.modern_chunk import CHUNK_SIZE
            # Convert world coordinates to local chunk coordinates
            local_x = int(world_x - chunk_x * CHUNK_SIZE)
            local_z = int(world_z - chunk_z * CHUNK_SIZE)
            chunk.set_block(local_x, int(world_y), local_z, block_type)
            return True
        
        return False
    
    def render_chunks(self, view_matrix=None, proj_matrix=None, camera_pos=None):
        """Render all loaded chunks with optional frustum and occlusion culling"""
        with self.thread_lock:
            chunks_to_render = list(self.chunks.items())
        
        total_chunks = len(chunks_to_render)
        frustum_culled = 0
        occlusion_culled = 0
        rendered_chunks = 0
        
        # Apply frustum culling first if enabled
        if self.enable_frustum_culling and view_matrix is not None and proj_matrix is not None:
            view_proj_matrix = proj_matrix * view_matrix
            self.frustum.extract_planes(view_proj_matrix)
            
            # Filter chunks using frustum culling
            frustum_visible_chunks = []
            for chunk_coords, chunk in chunks_to_render:
                chunk_x, chunk_z = chunk_coords
                if self.frustum.is_chunk_visible(chunk_x, chunk_z):
                    frustum_visible_chunks.append((chunk_coords, chunk))
                else:
                    frustum_culled += 1
            
            chunks_to_render = frustum_visible_chunks
        
        # Apply occlusion culling if enabled and camera position is available
        if self.enable_occlusion_culling and camera_pos is not None:
            # Filter chunks using occlusion culling
            occlusion_visible_chunks = self.occlusion_culler.filter_visible_chunks(camera_pos, chunks_to_render)
            occlusion_culled = len(chunks_to_render) - len(occlusion_visible_chunks)
            chunks_to_render = occlusion_visible_chunks
        
        # Render the remaining visible chunks (solid blocks first)
        for chunk_coords, chunk in chunks_to_render:
            chunk.render()
            rendered_chunks += 1
        
        # Then render transparent blocks (water, leaves) with proper blending
        for chunk_coords, chunk in chunks_to_render:
            if hasattr(chunk, 'render_transparent'):
                chunk.render_transparent()
        
        return rendered_chunks, total_chunks, frustum_culled, occlusion_culled
    
    def cleanup(self):
        """Clean up all chunks and resources"""
        print("ThreadedChunkManager cleanup...")
        
        # Stop background thread
        self.should_stop = True
        if self.loading_thread and self.loading_thread.is_alive():
            self.loading_thread.join(timeout=2.0)
        
        # Clean up all chunks
        with self.thread_lock:
            for chunk in self.chunks.values():
                if hasattr(chunk, 'vao') and chunk.vao:
                    chunk.vao.release()
            
            chunk_count = len(self.chunks)
            self.chunks.clear()
            self.loaded_chunks.clear()
        
        print(f"Cleaned up {chunk_count} chunks")
    
    def set_render_distance(self, new_distance):
        """Change the render distance and trigger chunk update"""
        if new_distance != self.render_distance:
            self.render_distance = new_distance
            # Force update on next frame
            self.last_player_chunk = None
            print(f"Render distance changed to: {new_distance}")
    
    def get_chunk_info(self):
        """Get information about loaded chunks for debugging"""
        with self.thread_lock:
            return {
                'loaded_chunks': len(self.chunks),
                'pending_chunks': len(self.loaded_chunks) - len(self.chunks),
                'cached_chunks': len(self.chunk_cache),
                'explored_chunks': len(self.explored_chunks),
                'queue_size': self.chunk_queue.qsize(),
                'completed_queue_size': self.completed_chunks.qsize(),
                'frustum_culling': self.enable_frustum_culling,
                'occlusion_culling': self.enable_occlusion_culling,
                'occlusion_cache_size': len(self.occlusion_culler.visibility_cache) if self.occlusion_culler else 0
            }
    
    def toggle_frustum_culling(self):
        """Toggle frustum culling on/off"""
        self.enable_frustum_culling = not self.enable_frustum_culling
        print(f"Frustum culling: {'enabled' if self.enable_frustum_culling else 'disabled'}")
        return self.enable_frustum_culling
    
    def toggle_occlusion_culling(self):
        """Toggle occlusion culling on/off"""
        self.enable_occlusion_culling = not self.enable_occlusion_culling
        if self.occlusion_culler:
            self.occlusion_culler.toggle_occlusion_culling()
        print(f"Occlusion culling: {'enabled' if self.enable_occlusion_culling else 'disabled'}")
        return self.enable_occlusion_culling
    
    def toggle_occlusion_conservative_mode(self):
        """Toggle conservative occlusion culling mode"""
        if self.occlusion_culler:
            # Check current threshold to determine mode
            is_conservative = self.occlusion_culler.occlusion_threshold >= 0.99
            self.occlusion_culler.set_conservative_mode(not is_conservative)
        return not is_conservative
