import math
import os
import glm
import moderngl as mgl
import numpy as np
import threading
import queue
import time
from world.modern_chunk import ModernChunk, CHUNK_SIZE
from world.blocks import FACE_LAYER, OPAQUE, SHAPE_TABLE
from world.fast_builder import build_chunk_mesh_fast, make_mesh_buffers, NO_NEIGHBOR
from engine.frustum import Frustum

HALF_CHUNK = CHUNK_SIZE / 2

# Where each LOD level takes over, in chunks from the player's chunk: level 0
# out to the first entry, level 1 out to the second, level 2 beyond — see
# build_chunk_mesh_fast for what each level gives up.
#
# Level 1 seals caves, which cannot be seen from outside the terrain — but it
# can be seen from *inside*, down a straight tunnel, and a player digging one
# would find it walled off. So the first ring is not set by what is visible
# from outside; it is set by how far anyone could see along a cave they are
# standing in. 128 blocks is past any natural cave (the carving noise turns
# over every ~11 blocks) and past most dug ones, and widening the ring from 48
# to 128 blocks costs 0.4 ms and 8% of the saving at render distance 16.
# ponytail: a hand-dug tunnel longer than 128 blocks still ends in a wall —
# gate sealing on the player being above ground if that ever comes up.
#
# Level 2 also drops per-corner AO, and that one is a real (if small) change,
# so it waits until a block is only a couple of pixels across. At 1200x800 and
# a 65 degree FOV a block covers about 628/d pixels, so at 16 chunks it is
# 2.5 px and an AO gradient one block wide has nowhere left to show. That also
# keeps AO everywhere at any render distance up to 16.
#
# Calibration, not structure: a wider FOV or a taller window makes a block
# cover more pixels and wants the second ring pushed out.
LOD_DISTANCES = (8, 16)
LOD_DISTANCES_SQ = tuple(d * d for d in LOD_DISTANCES)

class ThreadedChunkManager:
    """Manages dynamic chunk loading and unloading with background threading to eliminate lag"""

    # Worker threads. The terrain and meshing kernels are @njit(nogil=True), so
    # these genuinely run at the same time rather than taking turns on the GIL.
    # Deliberately only a slice of the machine: a game should not grab every
    # core it can see, and past four the main thread's VAO uploads become the
    # limit anyway.
    WORKER_COUNT = max(1, min(4, (os.cpu_count() or 2) - 2))

    # How long the main thread may spend folding finished chunks and meshes into
    # the world per frame. A fixed count alone is not enough: the workers can
    # produce faster than the main thread can upload, so a full batch of twelve
    # VAO uploads landing on one frame turned into a 25 ms hitch. A deadline
    # bounds that no matter how deep the backlog gets — the leftovers simply
    # arrive next frame.
    INTEGRATION_BUDGET = 0.003      # seconds

    
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
        self.worker_threads = []
        self.should_stop = False
        self.chunk_queue = queue.Queue()  # Queue for chunk operations
        self.completed_chunks = queue.Queue()  # Completed chunks ready for main thread
        self.chunks_to_unload = queue.Queue()  # Chunks to be unloaded
        self.thread_lock = threading.Lock()
        # chunk_cache and explored_chunks are the only structures the workers
        # really share with the main thread, and thread_lock never covered them.
        # self.chunks stays main-thread-only and needs no lock at all.
        self.cache_lock = threading.Lock()
        
        # Async mesh building queues
        self.mesh_build_queue = queue.PriorityQueue()  # Priority queue - closer chunks processed first
        self.completed_meshes = queue.Queue()  # Completed mesh data ready for VAO creation
        self.player_position = glm.vec3(0, 0, 0)  # Track player position for priority calculation
        self.mesh_request_counter = 0  # Counter for tiebreaker in priority queue
        
        # Frustum culling. The draw list and the coordinate arrays behind it are
        # rebuilt when the set of loaded chunks changes, not per frame — see
        # _refresh_draw_list.
        self.frustum = Frustum()
        self.enable_frustum_culling = True
        self._draw_chunks = []
        self._draw_list_dirty = True
        self._cull_mask = None

        self._update_fog()

        # Start background threads
        self.start_background_workers()

        print(f"ThreadedChunkManager initialized with render distance: {render_distance} "
              f"({self.WORKER_COUNT} worker threads)")

    
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
                self._draw_list_dirty = True
                self.loaded_chunks.add((x, z))
                self.explored_chunks.add((x, z))
                generated.append((x, z))

                print(f"Pre-generated chunk ({x}, {z}) - {len(generated)}/{self.chunks_to_pregenerate}")

            if len(generated) >= self.chunks_to_pregenerate:
                break

        # Mesh only after every chunk exists, so each one can see its neighbors
        # and skip the seam faces they cover.
        scratch = make_mesh_buffers()
        for chunk_x, chunk_z in generated:
            chunk = self.chunks[(chunk_x, chunk_z)]
            chunk.upload_mesh(*build_chunk_mesh_fast(
                chunk.blocks, chunk_x, chunk_z, self._neighbor_blocks(chunk_x, chunk_z),
                *scratch, FACE_LAYER, OPAQUE, SHAPE_TABLE))

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

        if not chunk.is_modified:
            with self.cache_lock:
                self.explored_chunks.add((chunk_x, chunk_z))
            return False

        data = chunk.save_chunk_data()
        with self.cache_lock:
            self.explored_chunks.add((chunk_x, chunk_z))
            self.chunk_cache[(chunk_x, chunk_z)] = data
        return True
    
    def load_chunk_from_cache(self, chunk_x, chunk_z):
        """Load chunk data from cache if available. Runs on a worker thread.

        The lock covers the lookup only — rebuilding the chunk copies 64 KB and
        several workers should be able to do that at the same time.
        """
        with self.cache_lock:
            chunk_data = self.chunk_cache.get((chunk_x, chunk_z))

        if chunk_data is None:
            return None
        return ModernChunk(chunk_x, chunk_z, self.renderer, chunk_data, chunk_manager=self)

    def is_chunk_explored(self, chunk_x, chunk_z):
        """Check if a chunk has been previously generated/explored"""
        with self.cache_lock:
            return (chunk_x, chunk_z) in self.explored_chunks
    
    def start_background_workers(self):
        """Start the background chunk loading and meshing threads"""
        for index in range(self.WORKER_COUNT):
            thread = threading.Thread(target=self._chunk_worker, daemon=True,
                                      name=f"chunk-worker-{index}")
            thread.start()
            self.worker_threads.append(thread)
    
    def _chunk_worker(self):
        """Background thread worker for chunk loading and mesh building"""
        # Owned by this thread for its whole life, so meshing allocates nothing
        # big and nothing is shared with the main thread's own set.
        scratch = make_mesh_buffers()

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
                            with self.cache_lock:
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
                    did_work = True

                    # Skip work whose result would only be thrown away: the
                    # chunk was unloaded while this request waited, or a newer
                    # request for it has already been queued. Both reads can be
                    # a moment stale, which costs at most one wasted build.
                    if (chunk_coords in self.loaded_chunks
                            and mesh_request['seq'] == chunk.mesh_seq):
                        # Build mesh in background thread. The neighbor arrays were
                        # picked up on the main thread when the request was queued.
                        mesh = build_chunk_mesh_fast(
                            chunk.blocks, chunk.chunk_x, chunk.chunk_z, mesh_request['neighbors'],
                            *scratch, FACE_LAYER, OPAQUE, SHAPE_TABLE,
                            mesh_request['lod']
                        )

                        # Queue completed mesh for main thread
                        self.completed_meshes.put({
                            'coords': chunk_coords,
                            'seq': mesh_request['seq'],
                            'mesh': mesh,
                        })
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

    def chunk_lod(self, chunk_x, chunk_z):
        """Detail level for this chunk, from its distance to the player's chunk.

        Same circular metric the loader uses, so the LOD rings are concentric
        with the render distance rather than square inside it.
        """
        if self.last_player_chunk is None:
            return 0

        dx = chunk_x - self.last_player_chunk[0]
        dz = chunk_z - self.last_player_chunk[1]
        distance_sq = dx * dx + dz * dz

        for level, limit_sq in enumerate(LOD_DISTANCES_SQ):
            if distance_sq <= limit_sq:
                return level
        return len(LOD_DISTANCES_SQ)

    def request_mesh(self, chunk_x, chunk_z, chunk=None):
        """Queue a background mesh build. No-op if the chunk is not loaded.

        The neighbor arrays are gathered here rather than in the worker because
        self.chunks is main-thread-only.
        """
        if chunk is None:
            chunk = self.chunks.get((chunk_x, chunk_z))
            if chunk is None:
                return

        # The level this chunk was last *asked* for, not the one it is showing.
        # update() compares against it to find chunks the player has walked into
        # a different ring of, so it has to move when the request goes out or
        # every frame would queue the same rebuild again.
        chunk.lod = self.chunk_lod(chunk_x, chunk_z)

        self.mesh_request_counter += 1
        # Stamp the chunk with this request's number. Workers finish out of
        # order, so anything coming back with an older stamp has been
        # superseded and must not overwrite the newer mesh.
        chunk.mesh_seq = self.mesh_request_counter

        self.mesh_build_queue.put((
            self._calculate_chunk_priority(chunk_x, chunk_z),
            self.mesh_request_counter,
            {
                'coords': (chunk_x, chunk_z),
                'chunk': chunk,
                'seq': self.mesh_request_counter,
                'lod': chunk.lod,
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
                
                # The same circular metric get_chunks_in_range uses, squared so
                # it needs no sqrt. This used to be a square, so it kept meshing
                # corner chunks that were never going to be loaded.
                chunk_dx = chunk_x - current_chunk_x
                chunk_dz = chunk_z - current_chunk_z

                if chunk_dx * chunk_dx + chunk_dz * chunk_dz <= self.render_distance ** 2:
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

    def _is_out_of_range(self, chunk_x, chunk_z):
        """True if this chunk sits outside the current render distance."""
        if self.last_player_chunk is None:
            return False

        dx = chunk_x - self.last_player_chunk[0]
        dz = chunk_z - self.last_player_chunk[1]
        return dx * dx + dz * dz > self.render_distance ** 2
    
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
        deadline = time.perf_counter() + self.INTEGRATION_BUDGET
        
        while processed < max_per_frame and time.perf_counter() < deadline:
            try:
                result = self.completed_chunks.get_nowait()
                if result['type'] == 'loaded':
                    chunk_x, chunk_z = result['coords']
                    chunk = result['chunk']

                    # It may have left the render distance while it was being
                    # generated. request_chunk_unload could not see it back then
                    # because it was not in self.chunks yet, so the request was
                    # dropped and the chunk stayed loaded until the next chunk
                    # crossing happened to notice. Let it go here instead.
                    if self._is_out_of_range(chunk_x, chunk_z):
                        self.loaded_chunks.discard((chunk_x, chunk_z))
                        processed += 1
                        continue

                    # Add chunk to loaded chunks now that it's ready (or mesh is building)
                    with self.thread_lock:
                        self.chunks[(chunk_x, chunk_z)] = chunk
                    self._draw_list_dirty = True

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
        while mesh_processed < max_per_frame and time.perf_counter() < deadline:
            try:
                mesh_data = self.completed_meshes.get_nowait()
                chunk_coords = mesh_data['coords']

                # Upload on the main thread (OpenGL requirement). A mesh whose
                # chunk was unloaded meanwhile is simply dropped.
                with self.thread_lock:
                    chunk = self.chunks.get(chunk_coords)

                # Drop results that a newer request has already superseded, and
                # results whose chunk has gone. Workers do not finish in the
                # order they were given work, so without the stamp an older
                # mesh could land on top of a newer one.
                if chunk is not None and mesh_data['seq'] == chunk.mesh_seq:
                    chunk.upload_mesh(*mesh_data['mesh'])

                # Counted either way, so a burst of stale results cannot blow
                # past the per-frame budget.
                mesh_processed += 1
            except queue.Empty:
                break
        
        # Process unload requests
        unload_count = 0
        while unload_count < max_per_frame and time.perf_counter() < deadline:
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
            self._draw_list_dirty = True
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

            # Chunks the player just walked into a different LOD ring of. Only
            # the two ring boundaries move, so this is a few dozen rebuilds even
            # at a large render distance, and they queue behind everything
            # closer because the priority is distance.
            # ponytail: no hysteresis — pacing back and forth across a boundary
            # re-meshes the same ring. Add a one-chunk deadband if it shows up
            # in a profile; the requests are already the lowest priority there is.
            lod_requests = 0
            for (chunk_x, chunk_z), chunk in self.chunks.items():
                if self.chunk_lod(chunk_x, chunk_z) != chunk.lod:
                    self.request_mesh(chunk_x, chunk_z, chunk)
                    lod_requests += 1

            if load_requests > 0 or unload_requests > 0 or lod_requests > 0:
                print(f"Player moved to chunk {current_chunk}. "
                      f"Load requests: {load_requests}, Unload requests: {unload_requests}, "
                      f"LOD rebuilds: {lod_requests}, Total chunks: {len(self.chunks)}")
        
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
    
    def _refresh_draw_list(self):
        """Rebuild the flat chunk list and the coordinate arrays culling runs on.

        Only the *set* of loaded chunks feeds these, so this runs when a chunk
        arrives or leaves rather than every frame. Rebuilding it per frame was
        how the old draw path started, and `list(self.chunks.items())` alone
        was 0.44 ms a frame at render distance 24.
        """
        with self.thread_lock:
            items = list(self.chunks.items())

        self._draw_chunks = [chunk for _, chunk in items]
        coords = (np.array([key for key, _ in items], dtype=np.float64)
                  if items else np.zeros((0, 2)))

        self._cull_min_x = coords[:, 0] * CHUNK_SIZE
        self._cull_max_x = self._cull_min_x + CHUNK_SIZE
        self._cull_min_z = coords[:, 1] * CHUNK_SIZE
        self._cull_max_z = self._cull_min_z + CHUNK_SIZE
        # Chunk centres, for the near-to-far sort.
        self._cull_mid_x = self._cull_min_x + HALF_CHUNK
        self._cull_mid_z = self._cull_min_z + HALF_CHUNK
        self._cull_mask = None
        self._draw_list_dirty = False

    def render_chunks(self, view_matrix=None, proj_matrix=None):
        """Render all loaded chunks with optional frustum culling"""
        if self._draw_list_dirty:
            self._refresh_draw_list()

        chunks = self._draw_chunks
        total_chunks = len(chunks)
        frustum_culled = 0

        if self.enable_frustum_culling and view_matrix is not None and proj_matrix is not None:
            self.frustum.extract_planes(proj_matrix * view_matrix)
            self._cull_mask = self.frustum.visible_mask(
                self._cull_min_x, self._cull_max_x,
                self._cull_min_z, self._cull_max_z, self._cull_mask)
            visible = np.flatnonzero(self._cull_mask)
            frustum_culled = total_chunks - len(visible)
        else:
            visible = np.arange(total_chunks)

        # Near to far, so the depth test can throw away fragments hidden behind
        # a closer hill before the fragment shader ever runs on them.
        dx = self._cull_mid_x[visible] - self.player_position.x
        dz = self._cull_mid_z[visible] - self.player_position.z
        order = visible[np.argsort(dx * dx + dz * dz)]

        # Straight to the VAO. This loop runs for every visible chunk on every
        # frame, and the two wrapper calls it used to go through did nothing but
        # forward.
        rendered_chunks = 0
        triangles = 0
        see_through = []
        for index in order.tolist():
            chunk = chunks[index]
            vao = chunk.vao
            if vao is not None:
                vao.render()
                rendered_chunks += 1
                triangles += chunk.vertex_count
            if chunk.vao_t is not None:
                see_through.append(chunk)

        # Second pass: glass, ice and the other see-through blocks, after every
        # opaque chunk, so there is something behind them to blend with. Far to
        # near — the reverse of the pass above, which was already sorted — so a
        # nearer pane blends over a farther one instead of hiding it.
        #
        # Depth writes stay ON. Turned off, panes inside one chunk blend in
        # mesher order, which is neither back-to-front nor stable as you walk;
        # left on, the worst case is a farther pane missing behind a nearer one,
        # and no flicker. Sorting quads per frame is the real fix and costs more
        # than this feature is worth (referans.md: TranslucencyPointOfView).
        #
        # Nothing generated is transparent, so on a fresh world this list is
        # empty and the whole block is one branch.
        if see_through:
            self.renderer.ctx.enable(mgl.BLEND)
            self.renderer.ctx.blend_func = mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA
            for chunk in reversed(see_through):
                chunk.vao_t.render()
                triangles += chunk.vertex_count_t
            self.renderer.ctx.disable(mgl.BLEND)

        return rendered_chunks, total_chunks, frustum_culled, triangles // 3

    def cleanup(self):
        """Clean up all chunks and resources"""
        print("ThreadedChunkManager cleanup...")
        
        # Stop background threads
        self.should_stop = True
        for thread in self.worker_threads:
            if thread.is_alive():
                thread.join(timeout=2.0)
        
        # Clean up all chunks
        with self.thread_lock:
            for chunk in self.chunks.values():
                chunk.release_gl()

            chunk_count = len(self.chunks)
            self.chunks.clear()
            self.loaded_chunks.clear()
        self._draw_list_dirty = True
        
        print(f"Cleaned up {chunk_count} chunks")
    
    def _update_fog(self):
        """Hand the renderer the radius the world is actually loaded to.

        It picks the fog start and end from that. Every render distance change
        routes through here (constructor and set_render_distance), so the shaders
        can never disagree with how far the world goes.
        """
        if self.renderer:
            self.renderer.set_fog_distance(self.render_distance * CHUNK_SIZE)

    def set_render_distance(self, new_distance):
        """Change the render distance and trigger chunk update"""
        if new_distance != self.render_distance:
            self.render_distance = new_distance
            self._update_fog()
            # Force update on next frame
            self.last_player_chunk = None
            print(f"Render distance changed to: {new_distance}")
    
    def get_chunk_info(self):
        """Get information about loaded chunks for debugging"""
        with self.thread_lock:
            lod_counts = [0] * (len(LOD_DISTANCES) + 1)
            for chunk in self.chunks.values():
                lod_counts[chunk.lod] += 1

            return {
                'lod_counts': lod_counts,
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
