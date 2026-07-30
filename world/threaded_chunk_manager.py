import math
import glm
import threading
import queue
import time
from world.modern_chunk import ModernChunk, CHUNK_SIZE
from world.fast_builder import build_chunk_mesh_fast, make_mesh_buffers, NO_NEIGHBOR
from engine.frustum import Frustum

class ThreadedChunkManager:
    """Manages dynamic chunk loading and unloading with background threading to eliminate lag"""
    
    def __init__(self, renderer, render_distance=8):
        self.renderer = renderer
        self.render_distance = render_distance
        self.chunks = {}  # Dictionary of (x, z) -> chunk (main thread access)
        self.loaded_chunks = set()  # Set of (x, z) coordinates for loaded chunks
        self.last_player_chunk = None
        self._offsets = []          # circular render-distance shape, see _range_offsets
        self._offsets_radius = None
        
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

        # Start background thread
        self.start_background_thread()
        
        print(f"ThreadedChunkManager initialized with render distance: {render_distance}")

    
    def pregenerate_spawn_chunks(self, spawn_x, spawn_z):
        """Pre-generate chunks around spawn position before game starts"""
        if self.initial_chunks_generated:
            return
        
        print(f"Pre-generating {self.chunks_to_pregenerate} chunks around spawn ({spawn_x}, {spawn_z})...")
        
        # Calculate spawn chunk coordinates
        spawn_chunk_x = int(spawn_x // CHUNK_SIZE)
        spawn_chunk_z = int(spawn_z // CHUNK_SIZE)
        
        # Generate chunks in a square around spawn
        radius = int(math.sqrt(self.chunks_to_pregenerate) // 2) + 1
        generated = []

        for x in range(spawn_chunk_x - radius, spawn_chunk_x + radius + 1):
            for z in range(spawn_chunk_z - radius, spawn_chunk_z + radius + 1):
                if len(generated) >= self.chunks_to_pregenerate:
                    break

                # Generate chunk immediately (synchronously for pre-gen)
                chunk = ModernChunk(x, z, self.renderer, chunk_data=None, chunk_manager=self)

                self.chunks[(x, z)] = chunk
                self.loaded_chunks.add((x, z))
                self.explored_chunks.add((x, z))
                generated.append((x, z))

                print(f"Pre-generated chunk ({x}, {z}) - {len(generated)}/{self.chunks_to_pregenerate}")

            if len(generated) >= self.chunks_to_pregenerate:
                break

        # Mesh only after every chunk exists, so each one can see its neighbors
        # and skip the seam faces they cover.
        scratch_vertices, scratch_indices = make_mesh_buffers()
        for chunk_x, chunk_z in generated:
            chunk = self.chunks[(chunk_x, chunk_z)]
            vertices, indices = build_chunk_mesh_fast(
                chunk.blocks, chunk_x, chunk_z, self._neighbor_blocks(chunk_x, chunk_z),
                scratch_vertices, scratch_indices)
            chunk.upload_mesh(vertices, indices)

        generated_count = len(generated)
        self.initial_chunks_generated = True
        print(f"Pre-generation complete! Generated {generated_count} chunks with meshes ready.")
    
    def save_chunk_to_cache(self, chunk_x, chunk_z):
        """Keep an unloaded chunk's blocks, but only if the player changed them.

        Terrain is a pure function of the chunk coordinates — fixed permutation
        table, hash-based per-column jitter — so an untouched chunk regenerates
        byte for byte in about 10 ms of worker time. Storing it instead meant
        holding 64 KB forever for something reproducible, and since the cache is
        never evicted, wandering around used to grow memory without bound.

        ponytail: edited chunks still accumulate with no ceiling. That is player
        work with nowhere else to live until chunks are written to disk; add a
        bounded LRU together with that, not before.
        """
        chunk = self.chunks.get((chunk_x, chunk_z))
        if chunk is None:
            return False

        self.explored_chunks.add((chunk_x, chunk_z))

        if not chunk.is_modified:
            return False

        self.chunk_cache[(chunk_x, chunk_z)] = chunk.save_chunk_data()
        return True
    
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
        # Owned by this thread for its whole life, so meshing allocates nothing
        # big and nothing is shared with the main thread's own set.
        scratch_vertices, scratch_indices = make_mesh_buffers()

        while not self.should_stop:
            try:
                did_work = False

                # Check for chunk loading requests
                try:
                    operation = self.chunk_queue.get_nowait()
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
                    did_work = True
                except queue.Empty:
                    pass

                # Check for mesh building requests (priority queue format)
                try:
                    priority_item = self.mesh_build_queue.get_nowait()
                    # PriorityQueue returns (priority, counter, data) tuple
                    priority, counter, mesh_request = priority_item
                    chunk_coords = mesh_request['coords']
                    chunk = mesh_request['chunk']
                    
                    # Build mesh in background thread. The neighbor arrays were
                    # picked up on the main thread when the request was queued.
                    vertices_array, indices_array = build_chunk_mesh_fast(
                        chunk.blocks, chunk.chunk_x, chunk.chunk_z, mesh_request['neighbors'],
                        scratch_vertices, scratch_indices
                    )
                    
                    # Queue completed mesh for main thread
                    self.completed_meshes.put({
                        'coords': chunk_coords,
                        'vertices': vertices_array,
                        'indices': indices_array
                    })
                    did_work = True
                except queue.Empty:
                    pass

                # Idle only when BOTH queues are empty. Blocking on the chunk
                # queue used to stall the worker 0.1 s per loop, capping mesh
                # output at ~10/sec no matter how deep the backlog was.
                # ponytail: 2 ms poll instead of a condition variable — swap in
                # threading.Event if this thread ever shows up in a profile.
                if not did_work:
                    time.sleep(0.002)

            except Exception as e:
                print(f"Error in chunk worker thread: {e}")
                time.sleep(0.1)
    
    def _calculate_chunk_priority(self, chunk_x, chunk_z):
        """Calculate priority for chunk based on distance to player (lower = higher priority)"""
        # Calculate chunk center world position
        chunk_center_x = (chunk_x * CHUNK_SIZE) + (CHUNK_SIZE / 2)
        chunk_center_z = (chunk_z * CHUNK_SIZE) + (CHUNK_SIZE / 2)
        
        # Calculate distance to player (use 2D distance, ignore Y)
        dx = chunk_center_x - self.player_position.x
        dz = chunk_center_z - self.player_position.z
        distance = math.sqrt(dx*dx + dz*dz)
        
        return distance  # Lower distance = higher priority

    def _neighbor_blocks(self, chunk_x, chunk_z):
        """The (-X, +X, -Z, +Z) neighbors' block arrays for the mesher.

        NO_NEIGHBOR fills in for chunks that are not loaded, which makes the
        mesher fall back to its old behaviour on that seam. Main thread only —
        it reads self.chunks, which the worker never touches.
        """
        return tuple(
            NO_NEIGHBOR if neighbor is None else neighbor.blocks
            for neighbor in (
                self.chunks.get((chunk_x - 1, chunk_z)),
                self.chunks.get((chunk_x + 1, chunk_z)),
                self.chunks.get((chunk_x, chunk_z - 1)),
                self.chunks.get((chunk_x, chunk_z + 1)),
            )
        )

    def request_mesh(self, chunk_x, chunk_z, chunk=None):
        """Queue a background mesh build. No-op if the chunk is not loaded.

        The neighbor arrays are gathered here rather than in the worker because
        self.chunks is main-thread-only.
        """
        if chunk is None:
            chunk = self.chunks.get((chunk_x, chunk_z))
            if chunk is None:
                return

        self.mesh_request_counter += 1
        self.mesh_build_queue.put((
            self._calculate_chunk_priority(chunk_x, chunk_z),
            self.mesh_request_counter,
            {
                'coords': (chunk_x, chunk_z),
                'chunk': chunk,
                'neighbors': self._neighbor_blocks(chunk_x, chunk_z),
            },
        ))

    def clear_distant_mesh_requests(self):
        """Clear mesh requests for chunks that are now too far from player"""
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
        chunk_x = int(world_x // CHUNK_SIZE)
        chunk_z = int(world_z // CHUNK_SIZE)
        return chunk_x, chunk_z
    
    def get_player_chunk(self, player_pos):
        """Get the chunk coordinates the player is currently in"""
        return self.world_to_chunk_coords(player_pos.x, player_pos.z)
    
    def _range_offsets(self):
        """Chunk offsets inside the circular render distance, cached per radius.

        The shape only depends on render_distance, so it is built once instead
        of on every chunk crossing — and compared squared, which drops the
        (2r+1)^2 square roots the old version did each time.
        """
        if self._offsets_radius != self.render_distance:
            radius = self.render_distance
            radius_sq = radius * radius
            self._offsets = [
                (dx, dz)
                for dx in range(-radius, radius + 1)
                for dz in range(-radius, radius + 1)
                if dx * dx + dz * dz <= radius_sq
            ]
            self._offsets_radius = radius

        return self._offsets

    def get_chunks_in_range(self, center_chunk_x, center_chunk_z):
        """Get all chunk coordinates within render distance of center chunk"""
        return {(center_chunk_x + dx, center_chunk_z + dz)
                for dx, dz in self._range_offsets()}
    
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

                    # Add chunk to loaded chunks now that it's ready (or mesh is building)
                    with self.thread_lock:
                        self.chunks[(chunk_x, chunk_z)] = chunk

                    self.request_mesh(chunk_x, chunk_z, chunk)

                    # This chunk now closes off the seam its neighbors were
                    # meshed against, so they have to drop those faces.
                    self.request_mesh(chunk_x - 1, chunk_z)
                    self.request_mesh(chunk_x + 1, chunk_z)
                    self.request_mesh(chunk_x, chunk_z - 1)
                    self.request_mesh(chunk_x, chunk_z + 1)

                    processed += 1

            except queue.Empty:
                break
        
        # Process completed meshes and create VAOs (main thread only for OpenGL)
        mesh_processed = 0
        while mesh_processed < max_per_frame:
            try:
                mesh_data = self.completed_meshes.get_nowait()
                chunk_coords = mesh_data['coords']

                # Upload on the main thread (OpenGL requirement). A mesh whose
                # chunk was unloaded meanwhile is simply dropped.
                with self.thread_lock:
                    chunk = self.chunks.get(chunk_coords)

                if chunk is not None:
                    chunk.upload_mesh(mesh_data['vertices'], mesh_data['indices'])

                # Counted either way, so a burst of stale results cannot blow
                # past the per-frame budget.
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
            chunk.release_gl()

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
        """Get a chunk at the given coordinates, or None if not loaded.

        No lock: self.chunks is only ever touched by the main thread. The worker
        hands finished chunks over through queues and never reaches into the
        dict, so taking the lock here bought nothing while costing an
        acquire/release on every one of the ~200 block queries the collision
        code and the raycast make each frame.
        """
        return self.chunks.get((chunk_x, chunk_z))
    
    def get_block_at(self, world_x, world_y, world_z):
        """Get block type at world coordinates.

        Hot path — the collision box sweep and the raycast call this a few
        hundred times per frame, so the helper calls are inlined here.
        """
        chunk_x = int(world_x // CHUNK_SIZE)
        chunk_z = int(world_z // CHUNK_SIZE)
        chunk = self.chunks.get((chunk_x, chunk_z))

        if chunk is None:
            return 0  # AIR if chunk not loaded

        # Convert world coordinates to local chunk coordinates
        local_x = int(world_x - chunk_x * CHUNK_SIZE)
        local_z = int(world_z - chunk_z * CHUNK_SIZE)
        return chunk.get_block(local_x, int(world_y), local_z)

    def set_block_at(self, world_x, world_y, world_z, block_type):
        """Set block type at world coordinates"""
        chunk_x, chunk_z = self.world_to_chunk_coords(world_x, world_z)
        chunk = self.get_chunk(chunk_x, chunk_z)

        if chunk:
            # Convert world coordinates to local chunk coordinates
            local_x = int(world_x - chunk_x * CHUNK_SIZE)
            local_z = int(world_z - chunk_z * CHUNK_SIZE)
            chunk.set_block(local_x, int(world_y), local_z, block_type)
            return True

        return False
    
    def render_chunks(self, view_matrix=None, proj_matrix=None):
        """Render all loaded chunks with optional frustum culling"""
        with self.thread_lock:
            chunks_to_render = list(self.chunks.items())

        total_chunks = len(chunks_to_render)
        frustum_culled = 0
        rendered_chunks = 0

        # Apply frustum culling if enabled
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

        # Render the remaining visible chunks (solid blocks first)
        for chunk_coords, chunk in chunks_to_render:
            chunk.render()
            rendered_chunks += 1
        
        # Then render transparent blocks (water, leaves) with proper blending
        for chunk_coords, chunk in chunks_to_render:
            if hasattr(chunk, 'render_transparent'):
                chunk.render_transparent()
        
        return rendered_chunks, total_chunks, frustum_culled

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
                chunk.release_gl()

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
                'frustum_culling': self.enable_frustum_culling
            }
    
    def toggle_frustum_culling(self):
        """Toggle frustum culling on/off"""
        self.enable_frustum_culling = not self.enable_frustum_culling
        print(f"Frustum culling: {'enabled' if self.enable_frustum_culling else 'disabled'}")
        return self.enable_frustum_culling
