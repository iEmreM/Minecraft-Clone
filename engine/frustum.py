import glm
import math

class Frustum:
    """Camera frustum for culling objects outside the view"""
    
    def __init__(self):
        # Frustum planes: left, right, bottom, top, near, far
        self.planes = [glm.vec4(0) for _ in range(6)]
        
    def extract_planes(self, view_projection_matrix):
        """Extract frustum planes from view-projection matrix"""
        m = view_projection_matrix
        
        # Left plane
        self.planes[0] = glm.vec4(
            m[0][3] + m[0][0],
            m[1][3] + m[1][0], 
            m[2][3] + m[2][0],
            m[3][3] + m[3][0]
        )
        
        # Right plane  
        self.planes[1] = glm.vec4(
            m[0][3] - m[0][0],
            m[1][3] - m[1][0],
            m[2][3] - m[2][0], 
            m[3][3] - m[3][0]
        )
        
        # Bottom plane
        self.planes[2] = glm.vec4(
            m[0][3] + m[0][1],
            m[1][3] + m[1][1],
            m[2][3] + m[2][1],
            m[3][3] + m[3][1]
        )
        
        # Top plane
        self.planes[3] = glm.vec4(
            m[0][3] - m[0][1],
            m[1][3] - m[1][1],
            m[2][3] - m[2][1],
            m[3][3] - m[3][1]
        )
        
        # Near plane
        self.planes[4] = glm.vec4(
            m[0][3] + m[0][2],
            m[1][3] + m[1][2],
            m[2][3] + m[2][2],
            m[3][3] + m[3][2]
        )
        
        # Far plane
        self.planes[5] = glm.vec4(
            m[0][3] - m[0][2],
            m[1][3] - m[1][2],
            m[2][3] - m[2][2],
            m[3][3] - m[3][2]
        )
        
        # Normalize all planes
        for i in range(6):
            length = math.sqrt(
                self.planes[i].x * self.planes[i].x +
                self.planes[i].y * self.planes[i].y +
                self.planes[i].z * self.planes[i].z
            )
            if length > 0:
                self.planes[i] /= length
    
    def is_aabb_inside(self, min_point, max_point):
        """Check if an axis-aligned bounding box is inside or intersecting the frustum"""
        for plane in self.planes:
            # Find the positive vertex (furthest point in direction of plane normal)
            positive_vertex = glm.vec3(
                max_point.x if plane.x >= 0 else min_point.x,
                max_point.y if plane.y >= 0 else min_point.y,
                max_point.z if plane.z >= 0 else min_point.z
            )
            
            # Calculate distance from positive vertex to plane
            distance = (plane.x * positive_vertex.x + 
                       plane.y * positive_vertex.y + 
                       plane.z * positive_vertex.z + 
                       plane.w)
            
            # If positive vertex is outside, the whole box is outside
            if distance < 0:
                return False
        
        return True
    
    def is_chunk_visible(self, chunk_x, chunk_z, chunk_size=16, chunk_height=256):
        """Check if a chunk is visible in the frustum"""
        # Calculate chunk world coordinates
        world_x = chunk_x * chunk_size
        world_z = chunk_z * chunk_size
        
        # Create bounding box for the chunk (full height)
        min_point = glm.vec3(world_x, 0, world_z)
        max_point = glm.vec3(world_x + chunk_size, chunk_height, world_z + chunk_size)
        
        return self.is_aabb_inside(min_point, max_point)
