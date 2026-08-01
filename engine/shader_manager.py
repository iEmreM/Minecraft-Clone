import re
import moderngl as mgl
from pathlib import Path


def load_source(path):
    """Read a shader, resolving `#include "file"` against its own directory.

    GLSL 330 has no #include, so it is resolved here. It exists for one file:
    sky_common.glsl, which sky.frag draws the sky with and chunk.frag /
    water.frag fog to — copies of that in three shaders is exactly how the fog
    and the sky drift apart.
    """
    path = Path(path)
    source = path.read_text()
    return re.sub(r'#include\s+"(.+?)"',
                  lambda m: path.with_name(m.group(1)).read_text(), source)


class ShaderManager:
    def __init__(self, ctx):
        self.ctx = ctx
        self.programs = {}
        
    def load_shader(self, name, vertex_path, fragment_path):
        """Load and compile a shader program"""
        try:
            # Create and compile shader program
            program = self.ctx.program(
                vertex_shader=load_source(vertex_path),
                fragment_shader=load_source(fragment_path)
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
        # Same vertex shader, same vertex format: the see-through blocks differ
        # only in that their fragment shader keeps the texture's alpha and
        # discards the fully clear texels. Keeping it a separate program is what
        # lets `chunk.frag` stay free of `discard`, which would cost the whole
        # terrain pass its early-Z.
        self.load_shader('chunk_alpha', 'shaders/chunk.vert', 'shaders/chunk_alpha.frag')
        self.load_shader('water', 'shaders/water.vert', 'shaders/water.frag')
