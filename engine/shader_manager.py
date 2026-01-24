import moderngl as mgl
from pathlib import Path


class ShaderManager:
    def __init__(self, ctx):
        self.ctx = ctx
        self.programs = {}
        
    def load_shader(self, name, vertex_path, fragment_path):
        """Load and compile a shader program"""
        try:
            # Read shader source files
            with open(vertex_path, 'r') as f:
                vertex_source = f.read()
            with open(fragment_path, 'r') as f:
                fragment_source = f.read()
            
            # Create and compile shader program
            program = self.ctx.program(
                vertex_shader=vertex_source,
                fragment_shader=fragment_source
            )
            
            self.programs[name] = program
            print(f"Loaded shader: {name}")
            return program
            
        except Exception as e:
            print(f"Error loading shader {name}: {e}")
            return None
    
    def get_program(self, name):
        """Get a compiled shader program"""
        return self.programs.get(name)
    
    def load_default_shaders(self):
        """Load the default shaders"""
        self.load_shader('chunk', 'shaders/chunk.vert', 'shaders/chunk.frag')
        self.load_shader('water', 'shaders/water.vert', 'shaders/water.frag')
