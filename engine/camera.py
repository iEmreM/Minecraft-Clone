import glm
import math


class Camera:
    # Skin width for the collision box. Large enough to absorb the float noise
    # left by per-axis resolution, small enough to be invisible at block scale.
    COLLISION_SKIN = 1e-3

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
        
        # Physics constants (scaled perfectly for real Minecraft values)
        self.sprinting = False
        self.gravity = -32.0  # Real Minecraft gravity is 32 m/s^2
        self.jump_velocity = 9.5
        self.walk_speed = 4.3
        self.sprint_speed = 7.0
        self.fly_speed = 10.9
        self.terminal_velocity = -78.4
        
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
        """Player AABB in world space, as {x1,x2,y1,y2,z1,z2} with x1 < x2 etc.

        *pos* is the eye position; the body hangs player_height below it. The box
        is inset on every side by COLLISION_SKIN so that resting flush against a
        surface does not read as an overlap. Without that inset, feet sitting
        exactly on a block top make floor(y1) land on the floor block itself, so
        every horizontal step collides with the ground the player is standing on
        — which is what snagged the player on the seam where two blocks meet.
        """
        half_width = self.player_width / 2
        skin = self.COLLISION_SKIN
        return {
            'x1': pos.x - half_width + skin,
            'x2': pos.x + half_width - skin,
            'y1': pos.y - self.player_height + skin,
            'y2': pos.y - skin,
            'z1': pos.z - half_width + skin,
            'z2': pos.z + half_width - skin
        }

    def _box_hits_block(self, bbox):
        """True if any non-air block overlaps *bbox*."""
        for block_x in range(int(math.floor(bbox['x1'])), int(math.floor(bbox['x2'])) + 1):
            for block_y in range(int(math.floor(bbox['y1'])), int(math.floor(bbox['y2'])) + 1):
                for block_z in range(int(math.floor(bbox['z1'])), int(math.floor(bbox['z2'])) + 1):
                    if self.world.get_block_at(block_x, block_y, block_z) != 0:
                        return True
        return False

    def check_collision_axis(self, old_pos, new_pos, axis):
        """Return the coordinate the player may actually occupy on *axis*.

        Each axis is resolved against old_pos independently, which is what lets
        the player slide along a wall instead of stopping dead against it.
        """
        if not self.world:
            return getattr(new_pos, axis)

        # Create test position with only this axis changed
        test_pos = glm.vec3(old_pos)
        setattr(test_pos, axis, getattr(new_pos, axis))

        if self._box_hits_block(self.get_bounding_box(test_pos)):
            return getattr(old_pos, axis)
        return getattr(new_pos, axis)

    def is_on_ground(self):
        """Check if player is standing on solid ground"""
        if not self.world:
            return False

        # Check slightly below feet
        test_pos = glm.vec3(self.position.x, self.position.y - self.ground_tolerance, self.position.z)
        bbox = self.get_bounding_box(test_pos)

        # Only check the Y level at the feet
        check_y = int(math.floor(bbox['y1']))
        for block_x in range(int(math.floor(bbox['x1'])), int(math.floor(bbox['x2'])) + 1):
            for block_z in range(int(math.floor(bbox['z1'])), int(math.floor(bbox['z2'])) + 1):
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
        if self.flying:
            speed = self.fly_speed * (2.0 if self.sprinting else 1.0)
        else:
            speed = self.sprint_speed if self.sprinting else self.walk_speed
            
        movement = glm.vec3(0.0)
        
        if self.strafe[0] != 0:  # Forward/backward
            # Use yaw to construct a robust horizontal forward vector for both walking and flying
            yaw_rad = glm.radians(self.yaw)
            horizontal_front = glm.vec3(math.cos(yaw_rad), 0.0, math.sin(yaw_rad))
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
        
        # Do not reset strafe here!
        # main.py resets strafe once per frame, enabling all 8
        # physical sub-steps to process horizontal velocity properly.
