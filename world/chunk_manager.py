import math
import glm
from world.modern_chunk import ModernChunk

class ChunkManager:
    """Manages dynamic chunk loading and unloading based on player position"""
    
    def __init__(self, renderer, render_distance=8):
        self.renderer = renderer
        self.render_distance = render_distance  # Chunks in each direction from player
        self.chunks = {}  # Dictionary of (x, z) -> chunk
        self.loaded_chunks = set()  # Set of (x, z) coordinates for loaded chunks
        self.last_player_chunk = None  # Last chunk position the player was in
        
        print(f"ChunkManager initialized with render distance: {render_distance}")
    
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
                # Optional: Use circular render distance instead of square
                distance = math.sqrt((x - center_chunk_x)**2 + (z - center_chunk_z)**2)
                if distance <= self.render_distance:
                    chunks_in_range.add((x, z))
        
        return chunks_in_range
    
    def load_chunk(self, chunk_x, chunk_z):
        """Load a chunk at the given coordinates"""
        if (chunk_x, chunk_z) not in self.chunks:
            # Create new chunk
            chunk = ModernChunk(chunk_x, chunk_z, self.renderer)
            self.chunks[(chunk_x, chunk_z)] = chunk
            self.loaded_chunks.add((chunk_x, chunk_z))
            return True
        return False
    
    def unload_chunk(self, chunk_x, chunk_z):
        """Unload a chunk at the given coordinates"""
        if (chunk_x, chunk_z) in self.chunks:
            chunk = self.chunks[(chunk_x, chunk_z)]
            
            # Clean up GPU resources
            if hasattr(chunk, 'vao') and chunk.vao:
                chunk.vao.release()
            
            # Remove from dictionaries
            del self.chunks[(chunk_x, chunk_z)]
            self.loaded_chunks.discard((chunk_x, chunk_z))
            return True
        return False
    
    def update(self, player_pos):
        """Update chunk loading/unloading based on player position"""
        current_chunk = self.get_player_chunk(player_pos)
        
        # Only update if player moved to a different chunk
        if current_chunk != self.last_player_chunk:
            self.last_player_chunk = current_chunk
            
            # Get chunks that should be loaded
            chunks_to_load = self.get_chunks_in_range(current_chunk[0], current_chunk[1])
            
            # Unload chunks that are too far away
            chunks_to_unload = []
            for chunk_coords in list(self.loaded_chunks):
                if chunk_coords not in chunks_to_load:
                    chunks_to_unload.append(chunk_coords)
            
            # Perform unloading
            unloaded_count = 0
            for chunk_x, chunk_z in chunks_to_unload:
                if self.unload_chunk(chunk_x, chunk_z):
                    unloaded_count += 1
            
            # Load new chunks
            loaded_count = 0
            for chunk_x, chunk_z in chunks_to_load:
                if self.load_chunk(chunk_x, chunk_z):
                    loaded_count += 1
            
            if loaded_count > 0 or unloaded_count > 0:
                print(f"Player moved to chunk {current_chunk}. "
                      f"Loaded: {loaded_count}, Unloaded: {unloaded_count}, "
                      f"Total chunks: {len(self.chunks)}")
    
    def get_chunk(self, chunk_x, chunk_z):
        """Get a chunk at the given coordinates, or None if not loaded"""
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
    
    def render_chunks(self):
        """Render all loaded chunks"""
        for chunk in self.chunks.values():
            chunk.render()
    
    def cleanup(self):
        """Clean up all chunks and resources"""
        print("ChunkManager cleanup...")
        for chunk in self.chunks.values():
            if hasattr(chunk, 'vao') and chunk.vao:
                chunk.vao.release()
        
        self.chunks.clear()
        self.loaded_chunks.clear()
        print(f"Cleaned up {len(self.chunks)} chunks")
    
    def set_render_distance(self, new_distance):
        """Change the render distance and trigger chunk update"""
        if new_distance != self.render_distance:
            self.render_distance = new_distance
            # Force update on next frame
            self.last_player_chunk = None
            print(f"Render distance changed to: {new_distance}")
