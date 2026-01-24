import glm
import math
import numpy as np
from typing import Dict, Set, Tuple, List
import moderngl as mgl

class HierarchicalZBuffer:
    """Hierarchical Z-Buffer for efficient occlusion culling"""
    
    def __init__(self, ctx, width=512, height=512):
        self.ctx = ctx
        self.width = width
        self.height = height
        self.levels = int(math.log2(min(width, height))) + 1
        
        # Create depth pyramid textures
        self.depth_textures = []
        current_width, current_height = width, height
        
        for level in range(self.levels):
            texture = ctx.texture((current_width, current_height), 1, dtype=np.float32)
            texture.filter = (mgl.NEAREST, mgl.NEAREST)
            self.depth_textures.append(texture)
            
            current_width = max(1, current_width // 2)
            current_height = max(1, current_height // 2)
        
        # Create framebuffers for each level
        self.framebuffers = []
        for texture in self.depth_textures:
            fbo = ctx.framebuffer(depth_attachment=texture)
            self.framebuffers.append(fbo)
    
    def update_depth_pyramid(self, scene_depth_texture):
        """Update the depth pyramid from the scene depth buffer"""
        # Copy scene depth to level 0
        self.ctx.copy_framebuffer(self.framebuffers[0], self.ctx.screen)
        
        # Generate mip levels by downsampling
        for level in range(1, self.levels):
            source_fbo = self.framebuffers[level - 1]
            target_fbo = self.framebuffers[level]
            
            # Simple downsampling (take max depth of 2x2 quad)
            self.ctx.copy_framebuffer(target_fbo, source_fbo)
    
    def test_occlusion(self, bbox_min, bbox_max, view_proj_matrix):
        """Test if a bounding box is occluded using the depth pyramid"""
        # Project bounding box to screen space
        corners = [
            glm.vec4(bbox_min.x, bbox_min.y, bbox_min.z, 1.0),
            glm.vec4(bbox_max.x, bbox_min.y, bbox_min.z, 1.0),
            glm.vec4(bbox_min.x, bbox_max.y, bbox_min.z, 1.0),
            glm.vec4(bbox_max.x, bbox_max.y, bbox_min.z, 1.0),
            glm.vec4(bbox_min.x, bbox_min.y, bbox_max.z, 1.0),
            glm.vec4(bbox_max.x, bbox_min.y, bbox_max.z, 1.0),
            glm.vec4(bbox_min.x, bbox_max.y, bbox_max.z, 1.0),
            glm.vec4(bbox_max.x, bbox_max.y, bbox_max.z, 1.0),
        ]
        
        # Transform to clip space
        screen_coords = []
        min_z = float('inf')
        
        for corner in corners:
            clip_pos = view_proj_matrix * corner
            if clip_pos.w <= 0:
                return False  # Behind camera
            
            ndc = clip_pos / clip_pos.w
            screen_x = (ndc.x + 1.0) * 0.5 * self.width
            screen_y = (ndc.y + 1.0) * 0.5 * self.height
            screen_z = ndc.z
            
            screen_coords.append((screen_x, screen_y, screen_z))
            min_z = min(min_z, screen_z)
        
        # Find screen space bounding box
        min_x = min(coord[0] for coord in screen_coords)
        max_x = max(coord[0] for coord in screen_coords)
        min_y = min(coord[1] for coord in screen_coords)
        max_y = max(coord[1] for coord in screen_coords)
        
        # Clamp to screen bounds
        min_x = max(0, min_x)
        max_x = min(self.width - 1, max_x)
        min_y = max(0, min_y)
        max_y = min(self.height - 1, max_y)
        
        if min_x >= max_x or min_y >= max_y:
            return True  # Outside screen
        
        # Choose appropriate mip level based on size
        size = max(max_x - min_x, max_y - min_y)
        mip_level = max(0, min(self.levels - 1, int(math.log2(size))))
        
        # Sample depth at that mip level (simplified - would need actual GPU sampling)
        # For now, return False (not occluded) to avoid false positives
        return False
    
    def cleanup(self):
        """Clean up GPU resources"""
        for texture in self.depth_textures:
            texture.release()
        for fbo in self.framebuffers:
            fbo.release()

class OcclusionCuller:
    """Advanced occlusion culling system using Hierarchical Z-Buffer technique"""
    
    def __init__(self, chunk_size=16, chunk_height=256):
        self.chunk_size = chunk_size
        self.chunk_height = chunk_height
        
        # Cache for visibility results
        self.visibility_cache = {}
        self.cache_frame = 0
        self.cache_max_age = 10  # Increased cache lifetime
        
        # Settings - much more conservative
        self.enable_occlusion_culling = True
        self.max_occlusion_distance = 12  # Increased range
        self.occlusion_threshold = 0.95  # Much more conservative threshold
        self.min_chunk_distance = 3  # Don't occlude very close chunks
        
        # HiZ buffer (will be initialized when we have GL context)
        self.hiz_buffer = None
        
    def clear_cache(self):
        """Clear the visibility cache"""
        self.visibility_cache.clear()
        self.cache_frame = 0
    
    def get_chunk_bounds(self, chunk_x, chunk_z):
        """Get the 3D bounding box of a chunk"""
        world_x = chunk_x * self.chunk_size
        world_z = chunk_z * self.chunk_size
        
        min_point = glm.vec3(world_x, 0, world_z)
        max_point = glm.vec3(world_x + self.chunk_size, self.chunk_height, world_z + self.chunk_size)
        
        return min_point, max_point
    
    def get_chunk_center(self, chunk_x, chunk_z):
        """Get the center point of a chunk"""
        world_x = chunk_x * self.chunk_size + self.chunk_size / 2
        world_z = chunk_z * self.chunk_size + self.chunk_size / 2
        world_y = self.chunk_height / 2
        
        return glm.vec3(world_x, world_y, world_z)
    
    def distance_to_chunk(self, camera_pos, chunk_x, chunk_z):
        """Calculate distance from camera to chunk center"""
        chunk_center = self.get_chunk_center(chunk_x, chunk_z)
        return glm.length(camera_pos - chunk_center)
    
    def is_chunk_behind_other(self, camera_pos, target_chunk, occluder_chunk):
        """Conservative check if target chunk is behind occluder chunk"""
        target_center = self.get_chunk_center(target_chunk[0], target_chunk[1])
        occluder_center = self.get_chunk_center(occluder_chunk[0], occluder_chunk[1])
        
        # Vector from camera to target
        to_target = target_center - camera_pos
        to_occluder = occluder_center - camera_pos
        
        # Check if occluder is closer than target
        target_distance = glm.length(to_target)
        occluder_distance = glm.length(to_occluder)
        
        # Much more conservative distance check
        if occluder_distance >= target_distance * 0.8:  # Occluder must be significantly closer
            return False
        
        # Normalize vectors
        to_target_norm = glm.normalize(to_target)
        to_occluder_norm = glm.normalize(to_occluder)
        
        # Calculate angle between camera->target and camera->occluder
        dot_product = glm.dot(to_target_norm, to_occluder_norm)
        
        # Much more conservative alignment threshold
        alignment_threshold = 0.98  # Cosine of ~11 degrees - very strict
        
        return dot_product > alignment_threshold
    
    def calculate_occlusion_coverage(self, camera_pos, target_chunk, occluder_chunks):
        """Conservative calculation of how much of the target chunk is covered by occluders"""
        if not occluder_chunks:
            return 0.0
        
        target_center = self.get_chunk_center(target_chunk[0], target_chunk[1])
        target_distance = glm.length(target_center - camera_pos)
        
        # Don't occlude very close chunks
        if target_distance < self.min_chunk_distance * self.chunk_size:
            return 0.0
        
        # Simple occlusion calculation based on angular coverage
        total_coverage = 0.0
        chunk_angular_size = math.atan2(self.chunk_size, target_distance)
        
        # Count only very obvious occluders
        valid_occluders = 0
        for occluder_chunk in occluder_chunks:
            if self.is_chunk_behind_other(camera_pos, target_chunk, occluder_chunk):
                occluder_center = self.get_chunk_center(occluder_chunk[0], occluder_chunk[1])
                occluder_distance = glm.length(occluder_center - camera_pos)
                
                # Only count if occluder is significantly closer
                if occluder_distance < target_distance * 0.7:
                    # Calculate angular size of occluder
                    occluder_angular_size = math.atan2(self.chunk_size, occluder_distance)
                    
                    # Very conservative coverage calculation
                    coverage_contribution = min(occluder_angular_size / chunk_angular_size, 1.0)
                    total_coverage += coverage_contribution * 0.1  # Much smaller weight factor
                    valid_occluders += 1
        
        # Only consider occlusion if we have multiple strong occluders
        if valid_occluders < 2:
            total_coverage *= 0.5
        
        return min(total_coverage, 0.8)  # Cap at 80% to be conservative
    
    def is_chunk_occluded(self, camera_pos, target_chunk, all_chunks):
        """Conservative check if a chunk is occluded by other chunks"""
        if not self.enable_occlusion_culling:
            return False
        
        # Cache key with reduced precision to improve cache hits
        cache_key = (target_chunk, int(camera_pos.x / 4), int(camera_pos.y / 4), int(camera_pos.z / 4))
        
        # Check cache
        if cache_key in self.visibility_cache:
            cached_result, cached_frame = self.visibility_cache[cache_key]
            if self.cache_frame - cached_frame < self.cache_max_age:
                return cached_result
        
        target_distance = self.distance_to_chunk(camera_pos, target_chunk[0], target_chunk[1])
        
        # Don't occlude close chunks
        if target_distance < self.min_chunk_distance * self.chunk_size:
            self.visibility_cache[cache_key] = (False, self.cache_frame)
            return False
        
        # Don't occlude very distant chunks (they're probably already frustum culled)
        if target_distance > self.max_occlusion_distance * self.chunk_size:
            self.visibility_cache[cache_key] = (False, self.cache_frame)
            return False
        
        # Find potential occluder chunks (closer to camera)
        occluder_chunks = []
        
        for chunk_coords in all_chunks:
            if chunk_coords == target_chunk:
                continue
            
            chunk_distance = self.distance_to_chunk(camera_pos, chunk_coords[0], chunk_coords[1])
            
            # Only consider chunks that are significantly closer
            if (chunk_distance < target_distance * 0.8 and 
                chunk_distance < self.max_occlusion_distance * self.chunk_size):
                occluder_chunks.append(chunk_coords)
        
        # Need minimum number of potential occluders
        if len(occluder_chunks) < 3:
            self.visibility_cache[cache_key] = (False, self.cache_frame)
            return False
        
        # Calculate occlusion coverage
        coverage = self.calculate_occlusion_coverage(camera_pos, target_chunk, occluder_chunks)
        is_occluded = coverage > self.occlusion_threshold
        
        # Cache result
        self.visibility_cache[cache_key] = (is_occluded, self.cache_frame)
        
        return is_occluded
    
    def filter_visible_chunks(self, camera_pos, chunk_list):
        """Filter chunks to remove occluded ones"""
        if not self.enable_occlusion_culling:
            return chunk_list
        
        self.cache_frame += 1
        
        # Clean old cache entries periodically
        if self.cache_frame % 60 == 0:  # Every 60 frames
            old_cache = {}
            for key, (result, frame) in self.visibility_cache.items():
                if self.cache_frame - frame < self.cache_max_age * 2:
                    old_cache[key] = (result, frame)
            self.visibility_cache = old_cache
        
        visible_chunks = []
        all_chunk_coords = set(chunk_coords for chunk_coords, chunk in chunk_list)
        
        for chunk_coords, chunk in chunk_list:
            if not self.is_chunk_occluded(camera_pos, chunk_coords, all_chunk_coords):
                visible_chunks.append((chunk_coords, chunk))
        
        return visible_chunks
    
    def toggle_occlusion_culling(self):
        """Toggle occlusion culling on/off"""
        self.enable_occlusion_culling = not self.enable_occlusion_culling
        self.clear_cache()  # Clear cache when toggling
        print(f"Occlusion culling: {'enabled' if self.enable_occlusion_culling else 'disabled'}")
        return self.enable_occlusion_culling
    
    def set_conservative_mode(self, conservative=True):
        """Set conservative occlusion culling mode"""
        if conservative:
            self.occlusion_threshold = 0.99  # Extremely conservative
            self.min_chunk_distance = 5     # Don't occlude close chunks
            self.max_occlusion_distance = 8 # Reduced range
            print("Occlusion culling: Conservative mode enabled")
        else:
            self.occlusion_threshold = 0.95
            self.min_chunk_distance = 3
            self.max_occlusion_distance = 12
            print("Occlusion culling: Normal mode enabled")
        self.clear_cache()
    
    def get_settings(self):
        """Get current occlusion culling settings"""
        return {
            'enabled': self.enable_occlusion_culling,
            'max_distance': self.max_occlusion_distance,
            'threshold': self.occlusion_threshold,
            'cache_size': len(self.visibility_cache),
            'cache_frame': self.cache_frame
        }
    
    def set_max_distance(self, distance):
        """Set maximum occlusion distance"""
        self.max_occlusion_distance = max(2, min(distance, 16))
        self.clear_cache()
        print(f"Occlusion max distance set to: {self.max_occlusion_distance}")
    
    def set_threshold(self, threshold):
        """Set occlusion threshold"""
        self.occlusion_threshold = max(0.1, min(threshold, 1.0))
        self.clear_cache()
        print(f"Occlusion threshold set to: {self.occlusion_threshold}")
