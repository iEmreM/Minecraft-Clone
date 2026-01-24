import glm
import math


class Camera:
    def __init__(self, position=(0, 0, 0), yaw=-90.0, pitch=-20.0):
        # Camera attributes
        self.position = glm.vec3(*position)
        self.yaw = yaw
        self.pitch = pitch
        
        # Camera vectors
        self.front = glm.vec3(0.0, 0.0, -1.0)
        self.up = glm.vec3(0.0, 1.0, 0.0)
        self.right = glm.vec3(1.0, 0.0, 0.0)
        self.world_up = glm.vec3(0.0, 1.0, 0.0)
        
        # Movement and physics
        self.flying = False
        self.velocity = glm.vec3(0.0, 0.0, 0.0)
        self.on_ground = False
        self.strafe = [0, 0]  # [forward/back, left/right]
        
        # Physics constants
        self.gravity = -30.0  # Stronger gravity like ornek1
        self.jump_velocity = 9.0
        self.walk_speed = 35.0
        self.fly_speed = 58.0  # Faster flying
        self.terminal_velocity = -50.0
        
        # Collision (like ornek1)
        self.world = None
        self.player_height = 1.8  # Like ornek1
        self.player_width = 0.4   # Like ornek1
        self.ground_tolerance = 0.01  # Small tolerance for ground detection
        
        # Update camera vectors
        self.update_camera_vectors()
    
    def get_view_matrix(self):
        """Calculate and return the view matrix"""
        return glm.lookAt(self.position, self.position + self.front, self.up)
    
    def set_world(self, world):
        """Set world reference for collision detection"""
        self.world = world
    
    def toggle_flying(self):
        """Toggle between flying and walking mode"""
        self.flying = not self.flying
        if self.flying:
            self.velocity.y = 0  # Stop falling when entering fly mode
        print(f"Flying mode: {'ON' if self.flying else 'OFF'}")
        return self.flying
    
    def jump(self):
        """Jump if on ground and not flying"""
        if not self.flying and self.on_ground:
            self.velocity.y = self.jump_velocity
            self.on_ground = False
    
    def update_camera_vectors(self):
        """Calculate front vector from yaw and pitch"""
        front = glm.vec3()
        front.x = math.cos(glm.radians(self.yaw)) * math.cos(glm.radians(self.pitch))
        front.y = math.sin(glm.radians(self.pitch))
        front.z = math.sin(glm.radians(self.yaw)) * math.cos(glm.radians(self.pitch))
        
        self.front = glm.normalize(front)
        self.right = glm.normalize(glm.cross(self.front, self.world_up))
        self.up = glm.normalize(glm.cross(self.right, self.front))
    
    def process_mouse_movement(self, xoffset, yoffset, constrain_pitch=True):
        """Process mouse movement input"""
        sensitivity = 0.2
        xoffset *= sensitivity
        yoffset *= sensitivity
        
        self.yaw += xoffset
        self.pitch += yoffset
        
        # Constrain pitch to avoid screen flip
        if constrain_pitch:
            self.pitch = max(-89.0, min(89.0, self.pitch))
        
        self.update_camera_vectors()
    
    def get_bounding_box(self, pos):
        """Get player bounding box"""
        half_width = self.player_width / 2
        half_height = self.player_height / 2
        return {
            'x1': pos.x - half_width,
            'x2': pos.x - self.player_width*2,
            'y1': pos.y - self.player_height,
            'y2': pos.y - half_height,
            'z1': pos.z - half_width,
            'z2': pos.z - self.player_width*2
        }
    
    def check_collision_axis(self, old_pos, new_pos, axis):
        """Check collision along a specific axis"""
        if not self.world:
            return new_pos
        
        # Create test position with only this axis changed
        test_pos = glm.vec3(old_pos)
        if axis == 'x':
            test_pos.x = new_pos.x
        elif axis == 'y':
            test_pos.y = new_pos.y
        elif axis == 'z':
            test_pos.z = new_pos.z
        
        bbox = self.get_bounding_box(test_pos)
        
        # Check all blocks in bounding box range
        min_x = int(math.floor(bbox['x1']))
        max_x = int(math.ceil(bbox['x2']))
        min_y = int(math.floor(bbox['y1']))
        max_y = int(math.ceil(bbox['y2']))
        min_z = int(math.floor(bbox['z1']))
        max_z = int(math.ceil(bbox['z2']))
        
        for block_x in range(min_x, max_x + 1):
            for block_y in range(min_y, max_y + 1):
                for block_z in range(min_z, max_z + 1):
                    if self.world.get_block_at(block_x, block_y, block_z) != 0:
                        # Block collision detected
                        if axis == 'x':
                            return old_pos.x
                        elif axis == 'y':
                            return old_pos.y
                        elif axis == 'z':
                            return old_pos.z
        
        # No collision
        if axis == 'x':
            return new_pos.x
        elif axis == 'y':
            return new_pos.y
        elif axis == 'z':
            return new_pos.z
    
    def is_on_ground(self):
        """Check if player is standing on solid ground"""
        if not self.world:
            return False
        
        # Check slightly below feet
        test_pos = glm.vec3(self.position.x, self.position.y - self.ground_tolerance, self.position.z)
        bbox = self.get_bounding_box(test_pos)
        
        # Only check Y level at feet
        check_y = int(math.floor(bbox['y1']))
        min_x = int(math.floor(bbox['x1']))
        max_x = int(math.ceil(bbox['x2']))
        min_z = int(math.floor(bbox['z1']))
        max_z = int(math.ceil(bbox['z2']))
        
        for block_x in range(min_x, max_x + 1):
            for block_z in range(min_z, max_z + 1):
                if self.world.get_block_at(block_x, check_y, block_z) != 0:
                    return True
        
        return False
    
    def update_physics(self, delta_time):
        """Update physics with proper collision detection"""
        old_pos = glm.vec3(self.position)
        
        # Apply gravity
        if not self.flying:
            self.velocity.y += self.gravity * delta_time
            if self.velocity.y < self.terminal_velocity:
                self.velocity.y = self.terminal_velocity
        else:
            self.velocity.y = 0
        
        # Calculate horizontal movement
        speed = self.fly_speed if self.flying else self.walk_speed
        movement = glm.vec3(0.0)
        
        if self.strafe[0] != 0:  # Forward/backward
            if self.flying:
                movement += self.front * self.strafe[0] * speed * delta_time
            else:
                # For walking, use horizontal component only
                horizontal_front = glm.normalize(glm.vec3(self.front.x, 0.0, self.front.z))
                movement += horizontal_front * self.strafe[0] * speed * delta_time
        
        if self.strafe[1] != 0:  # Left/right
            movement += self.right * self.strafe[1] * speed * delta_time
        
        # Calculate new position
        new_pos = glm.vec3(
            old_pos.x + movement.x,
            old_pos.y + (self.velocity.y * delta_time if not self.flying else 0),
            old_pos.z + movement.z
        )
        
        # Test collision for each axis separately
        final_x = self.check_collision_axis(old_pos, new_pos, 'x')
        final_z = self.check_collision_axis(old_pos, new_pos, 'z')
        
        # Apply horizontal movement
        self.position.x = final_x
        self.position.z = final_z
        
        # Handle vertical movement (gravity/jumping)
        if not self.flying:
            final_y = self.check_collision_axis(old_pos, new_pos, 'y')
            
            if final_y != new_pos.y:  # Collision in Y
                if self.velocity.y < 0:  # Hit ground
                    self.velocity.y = 0
                    self.on_ground = True
                elif self.velocity.y > 0:  # Hit ceiling
                    self.velocity.y = 0
                    self.on_ground = False
                
                self.position.y = final_y
            else:
                # No Y collision, apply movement
                self.position.y = final_y
                self.on_ground = self.is_on_ground()
        
        # Reset strafe
        self.strafe = [0, 0]
    
    def process_keyboard(self, direction, velocity):
        """Simple movement for flying mode vertical movement"""
        if self.flying:
            if direction == "UP":
                self.position.y += velocity
            elif direction == "DOWN":
                self.position.y -= velocity
